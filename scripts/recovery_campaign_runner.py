"""Future-only one-unit ledger adapter. No command-line launch entry point.

The caller supplies a reviewed frozen study, explicit next index and owned
SupervisedWork context. No automatic campaign launch or seed choice occurs here.
"""
import json
from pathlib import Path
import sqlite3
import time
from scripts.recovery_campaign_plan import build,bind_request
from src.experiments.execution_journal import digest
from src.experiments.recovery_execution import execute_attempt


def committed_dependencies(root,plan,unit):
    """Called only inside execute_attempt's ownership lock after reservation."""
    root = Path(root).resolve()
    db = sqlite3.connect((root/'attempts.sqlite').as_uri()+'?mode=ro',uri=True)
    try:
        saved = db.execute('SELECT value FROM plan').fetchall()
        if len(saved)!=1 or json.loads(saved[0][0])!=plan:
            raise ValueError('ledger plan identity mismatch')
        rows = db.execute("SELECT unit,artifact,sha256 FROM attempts WHERE status='complete' ORDER BY id").fetchall()
    finally:
        db.close()
    if [row[0] for row in rows] != list(range(len(rows))):
        raise ValueError('committed units must form an ordered prefix')
    completed = {plan['units'][index]['item']:(index,path,checksum) for index,path,checksum in rows}
    records = []
    for item in unit['specification']['dependencies']:
        if item not in completed:
            raise ValueError('scientific dependency is not committed')
        index,relative,checksum = completed[item]
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or digest(path)!=checksum:
            raise ValueError('committed dependency path/hash mismatch')
        records.append(dict(item=item,path=str(path),sha256=checksum,
                            input_identity=plan['units'][index]['input_identity']))
    return records


def execute_unit(root,study,index,worker,started_at=None,boundary=lambda event:None,
                 preflight=lambda:None,close_worker=False):
    began = time.perf_counter() if started_at is None else started_at
    plan = build(study)
    if type(index) is not int or not 0<=index<len(plan['units']):
        raise ValueError('valid explicit next unit index required')
    unit = plan['units'][index]
    # Per-worker invocation only, never persisted across process restarts.
    # Every consumed dependency is still read and hashed by parent and worker.
    invocation_key=(str(Path(root).resolve()),plan['identity'])
    initialized=getattr(worker,'validated_campaign',None)==invocation_key
    required=set(unit['specification']['dependencies']) if initialized else None
    def validate():
        checked=preflight() or {}
        if study['callback'] != dict(module='scripts.recovery_campaign_dispatch',function='dispatch'):
            raise ValueError('fixed scientific dispatcher required')
        for path,checksum in study['source_hashes'].items():
            if checked.get(path)==checksum:
                continue
            if digest(path)!=checksum:
                raise ValueError(f'frozen source changed: {path}')
        for name in ('calibration','ordering'):
            record = study['inputs'][name]
            if digest(record['path'])!=record['sha256']:
                raise ValueError(f'frozen input changed: {name}')
    def work(budget):
        dependencies = committed_dependencies(root,plan,unit)
        request = bind_request(study,unit,dependencies)
        return worker.call(study['callback']['module'],study['callback']['function'],request,budget)
    result=execute_attempt(root,plan['identity'],plan['units'],plan['caps'],unit['item'],
                           unit['input_identity'],work,validate,started_at=began,boundary=boundary,
                           committed_items=required,finalize=worker.close if close_worker else lambda:None)
    worker.validated_campaign=None if close_worker else invocation_key
    return result


def execute_launch_unit(root,study,index,worker,protocol,runtime,*,started_at=None,last=False):
    """Launch-facing path: mandatory cache/source/runtime checks under reservation.

Caller must freeze protocol/runtime and provide invocation-entry timing on its
first unit, previous-unit handoff timing thereafter, and last=True on batch end.
No CLI launch or implicit fresh namespace is provided here.
"""
    from scripts.recovery_source_inventory import validate_coverage,runtime_identity
    def preflight():
        checked=validate_coverage(Path.cwd(),study,protocol)
        if runtime_identity()!=runtime:
            raise ValueError('runtime differs from frozen invocation identity')
        return checked
    return execute_unit(root,study,index,worker,started_at=started_at,preflight=preflight,close_worker=last)
