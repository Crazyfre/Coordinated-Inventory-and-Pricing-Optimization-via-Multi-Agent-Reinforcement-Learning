"""Future-only owned attempts with durable outputs and cooperative limits.

Not a scientific runner or a preemptive watchdog. Call before_step before EVERY
simulator step, including deterministic reconstruction. No old archive migration.
"""
import time
import traceback
from pathlib import Path
from src.experiments.execution_guard import RunLock
from src.experiments.execution_journal import digest
from src.experiments.recovery_json import atomic_json
from src.experiments.recovery_ledger import RecoveryLedger


class StepBudget:
    def __init__(self, days, deadline, clock):
        self.maximum=days;self.deadline=deadline;self.clock=clock;self.days=0

    def check(self):
        if self.clock()>self.deadline:raise TimeoutError('attempt deadline exceeded')

    def before_step(self):
        self.check()
        if self.days>=self.maximum:raise RuntimeError('simulator day reservation exhausted')
        self.days+=1


def execute_attempt(root, identity, units, caps, item, input_identity, work,
                    validate_inputs, clock=time.perf_counter, started_at=None,
                    boundary=lambda event:None, committed_items=None, finalize=lambda:None):
    """work(budget) returns JSON payload; boundary is a crash-test hook only.

Output writes are owned here, not supplied by work. Startup/hash validation is
charged from entry. Final ledger receipt/lock release cannot time themselves.
committed_items=None checks the full prefix; an explicit set checks those
dependencies only. Scientific callers must also perform a full startup and
final archive check; this option never substitutes cached hashes for reads.
"""
    began=clock() if started_at is None else started_at
    root=Path(root).resolve()
    with RunLock(root.with_name(root.name+'.lock')):
        if root.exists() and not (root/'attempts.sqlite').is_file():
            raise ValueError('existing directory without recovery ledger; migration forbidden')
        root.mkdir(parents=True,exist_ok=True)
        ledger=RecoveryLedger(root/'attempts.sqlite',identity,units,caps)
        try:
            ledger.recover() # OS ownership acquired; previous worker cannot own it.
            attempt=ledger.begin(item,input_identity)
            unit=units[next(r['unit'] for r in ledger.attempts() if r['id']==attempt)]
            budget=StepBudget(unit['days'],began+unit['seconds'],clock)
            finalized=False
            stage='reservation_boundary'
            try:
                boundary('reserved')
                budget.check()
                stage='committed_artifact_validation'
                checked_items=set()
                for row in ledger.attempts():
                    if row['status']=='complete':
                        prior_item=units[row['unit']]['item']
                        if committed_items is not None and prior_item not in committed_items:
                            continue
                        path=root/row['artifact']
                        if not path.resolve().is_relative_to(root) or digest(path)!=row['sha256']:
                            raise ValueError('committed artifact corruption')
                        budget.check()
                        checked_items.add(prior_item)
                if committed_items is not None and checked_items!=set(committed_items):
                    raise ValueError('required committed dependency missing')
                stage='input_validation'
                budget.check();validate_inputs();budget.check()
                stage='work'
                payload=work(budget);budget.check()
                stage='artifact_write'
                path=root/f'attempt_{attempt:06d}.json'
                artifact=dict(item=item,input_identity=input_identity,days=budget.days,payload=payload)
                if hasattr(budget,'worker_receipt'):
                    artifact['worker_receipt']=budget.worker_receipt
                atomic_json(path,artifact)
                boundary('artifact_written')
                stage='artifact_hash'
                checksum=digest(path);budget.check()
                stage='worker_finalization'
                finalize();finalized=True;budget.check()
                stage='ledger_commit'
                settled_at=clock()
                ledger.finish(attempt,budget.days,settled_at-began,path.name,checksum)
            except BaseException:
                failure_traceback=traceback.format_exc()
                cleanup_error=None
                if not finalized:
                    try:finalize()
                    except BaseException as exc:cleanup_error=f'{type(exc).__name__}: {exc}'
                # Deadline termination has latency; do not hide observed overruns.
                observed=clock()-began
                # If a completion validation already failed, preserve its state.
                row=next(r for r in ledger.attempts() if r['id']==attempt)
                if row['status']=='running':
                    ledger.finish(attempt,unit['days'],unit['seconds'],'failed','0'*64,failed=True)
                atomic_json(root/f'failure_{attempt:06d}.json',dict(
                    attempt=attempt,observed_seconds=observed,reserved_seconds=unit['seconds'],
                    cleanup_error=cleanup_error,
                    stage=stage,traceback=failure_traceback,
                    parent_step_admissions=budget.days,
                    step_count_scope='parent budget only; remote worker steps may be unknown',
                    observed_overrun_seconds=max(0.,observed-unit['seconds']),
                    scope='entry through caught failure/worker cleanup; excludes receipt persistence'))
                raise
            boundary('committed')
            return dict(attempt=attempt,artifact=str(path),days=budget.days,charged=ledger.totals(),
                        settled_at=settled_at)
        finally:ledger.close()
