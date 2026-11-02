"""Prospective invocation wrapper, explicit frozen declarations only; no CLI.

Control metadata lives beside the ledger root so an empty legacy directory is
never mistaken for a fresh campaign. Caller records entry before imports.
"""
import json
from pathlib import Path
import sqlite3
import time
from scripts.recovery_campaign_plan import build
from scripts.recovery_campaign_runner import execute_launch_unit
from src.experiments.execution_guard import RunLock
from src.experiments.execution_journal import atomic_json
from src.experiments.supervised_work import SupervisedWork
from scripts.recovery_overhead import charges as overhead_charges
from scripts.recovery_overhead import combined_admission


def validate_schedule(plan,schedule):
    cursor=0
    for span in schedule:
        if (not isinstance(span,list) or len(span)!=2 or
                any(type(v) is not int for v in span) or span[0]!=cursor or span[1]<=span[0]):
            raise ValueError('contiguous nonempty half-open invocation spans required')
        cursor=span[1]
    if cursor!=len(plan['units']):
        raise ValueError('schedule must cover the complete scientific plan')


def completed_prefix(root,plan):
    path=Path(root)/'attempts.sqlite'
    if not path.exists():return 0
    db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        saved=db.execute('SELECT value FROM plan').fetchall()
        if len(saved)!=1 or json.loads(saved[0][0])!=plan:
            raise ValueError('invocation ledger plan mismatch')
        rows=db.execute('SELECT unit,status FROM attempts ORDER BY id').fetchall()
    finally:db.close()
    if any(status=='failed' for _,status in rows):
        raise RuntimeError('terminal failed scientific attempt')
    done=[unit for unit,status in rows if status=='complete']
    if done!=list(range(len(done))):raise ValueError('invalid completed prefix')
    return len(done)


def attempt_count(root):
    path=Path(root)/'attempts.sqlite'
    if not path.exists():return 0
    db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
    try:return db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
    finally:db.close()


def run_invocation(root,study,protocol,runtime,schedule,started_at=None,*,overhead=None):
    began=time.perf_counter() if started_at is None else started_at
    root=Path(root).resolve()
    control=root.with_name(root.name+'_invocations')
    # Separate invocation ownership spans between-unit gaps as well as work.
    with RunLock(root.with_name(root.name+'.invocation.lock')):
        control.mkdir(parents=True,exist_ok=True)
        plan=build(study);validate_schedule(plan,schedule)
        declaration=dict(study=study,protocol=protocol,runtime=runtime,schedule=schedule)
        if overhead is not None:declaration['overhead']=overhead
        manifest=control/'declaration.json'
        if manifest.exists():
            if json.loads(manifest.read_text())!=declaration:
                raise ValueError('frozen invocation declaration mismatch')
        else:atomic_json(manifest,declaration)
        old=sorted(control.glob('invocation_*.json'))
        for path in old:
            receipt=json.loads(path.read_text())
            if receipt['status']=='failed':raise RuntimeError('terminal invocation failure')
            if receipt['status']=='running':
                if (type(receipt.get('attempts_before')) is not int or
                        type(receipt.get('completed_before')) is not int or 'span' not in receipt):
                    receipt.update(status='failed',elapsed_seconds=None,
                        error='interruption before durable invocation boundaries; continuation forbidden')
                    atomic_json(path,receipt)
                    raise RuntimeError('interrupted invocation lacks durable boundaries')
                receipt.update(status='interrupted',elapsed_seconds=None)
                atomic_json(path,receipt)
        number=len(old)+1;receipt_path=control/f'invocation_{number:06d}.json'
        receipt=dict(number=number,status='running',started_wall=time.time(),
                     completed_before=None,completed_after=None,elapsed_seconds=None)
        atomic_json(receipt_path,receipt)
        worker=SupervisedWork(control/'scratch')
        settled=began;completed=None
        try:
            receipt['attempts_before']=attempt_count(root)
            completed=completed_prefix(root,plan)
            receipt['completed_before']=completed
            # A final commit can precede a parent crash/receipt write. Reconcile
            # without starting another scientific attempt or inventing elapsed.
            span=([completed,completed] if completed==len(plan['units']) else
                  next(s for s in schedule if s[0]<=completed<s[1]))
            receipt['span']=span;atomic_json(receipt_path,receipt)
            previous=[json.loads(path.read_text()) for path in old]
            if overhead is not None:
                reserved_overhead=overhead_charges(previous+[receipt],overhead,plan)
                combined_admission(root,plan,reserved_overhead)
            for index in range(completed,span[1]):
                if overhead is not None:
                    combined_admission(root,plan,reserved_overhead,index)
                result=execute_launch_unit(root,study,index,worker,protocol,runtime,
                    started_at=settled,last=index==span[1]-1)
                settled=result['settled_at']
                completed=index+1
            worker.close()
            final_attempts=attempt_count(root)
            ended=time.perf_counter()
            receipt.update(status='complete',completed_after=completed,elapsed_seconds=ended-began,
                           trailing_seconds=ended-settled,attempts_after=final_attempts)
            if overhead is not None:
                combined_admission(root,plan,overhead_charges(previous+[receipt],overhead,plan))
            atomic_json(receipt_path,receipt)
            return receipt
        except BaseException as exc:
            cleanup_error=None
            try:worker.close()
            except BaseException as cleanup:cleanup_error=f'{type(cleanup).__name__}: {cleanup}'
            receipt.update(status='failed',completed_after=completed,elapsed_seconds=time.perf_counter()-began,
                           error=f'{type(exc).__name__}: {exc}',cleanup_error=cleanup_error)
            atomic_json(receipt_path,receipt)
            raise


def launch(root,declaration,started_at=None):
    """Scientific entry API: overhead policy is mandatory, never a cap increase."""
    if set(declaration)!={'study','protocol','runtime','schedule','overhead'}:
        raise ValueError('complete launch declaration including overhead required')
    from scripts.recovery_overhead import validate
    validate(declaration['overhead'])
    return run_invocation(root,**declaration,started_at=started_at)
