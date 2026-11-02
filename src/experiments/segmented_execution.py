"""Single declared work unit, joining ownership, budgets and artifact commit.

Scientific callers supply the plan and work; no simulator or policy code is
imported here. A failed/ambiguous unit is terminal, never automatically retried.
"""
import time
from pathlib import Path
from src.experiments.execution_journal import ExecutionJournal
from src.experiments.execution_guard import RunLock,CumulativeClock


def execute_one(root,identity,items,maximum_days,caps,item,phase,days,work,
                clock=time.perf_counter,started_at=None):
    """work(artifact_path, deadline) writes its artifact and returns actual days.

    Caller supplies started_at from its entry point to include its own startup.
    The callback must enforce deadline during long-running scientific work.
    Late completion is also rejected here; this is not a preemptive watchdog.
    """
    began=clock() if started_at is None else started_at
    root=Path(root)
    # Lock outside the journal root: journal creation must remain exclusive.
    with RunLock(root.with_name(root.name+'.lock')):
        journal=ExecutionJournal(root,identity,items,maximum_days)
        timer=CumulativeClock(root/'time.json',identity,caps,clock)
        committed=[r['item'] for r in journal.state['completed']]
        if committed!=timer.state['completed']:
            raise RuntimeError('journal/time commit mismatch; continuation forbidden')
        if len(committed)>=len(items) or item!=items[len(committed)]:
            raise ValueError('next declared unit required')
        deadline=timer.begin(item,phase,started_at=began)
        try:
            # Time reservation precedes day reservation: either pending marker
            # makes interruption detectable at every later crash boundary.
            journal.claim(item,days)
            if clock()>deadline:raise TimeoutError('budget exhausted during initialization')
            artifact=root/f'unit_{len(committed):04d}.json'
            if artifact.exists():raise RuntimeError('uncommitted artifact exists')
            actual=work(artifact,deadline)
            if clock()>deadline:raise TimeoutError('unit exceeded cumulative deadline')
            journal.commit(artifact,actual)
        except BaseException as exc:
            timer.finish(error=f'{type(exc).__name__}: {exc}')
            raise
        # A crash after journal commit but before this commit leaves the clock
        # pending and therefore cannot incorrectly skip or duplicate the unit.
        timer.finish()
        return dict(item=item,artifact=str(artifact),days=actual,
                    total_days=journal.state['reserved_days'],used_seconds=timer.state['used'].copy())
