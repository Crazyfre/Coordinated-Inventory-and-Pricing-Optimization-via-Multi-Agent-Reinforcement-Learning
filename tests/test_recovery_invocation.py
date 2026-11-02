import json
import time
import pytest
from scripts import recovery_invocation as invocation
from scripts.recovery_campaign_plan import build
from tests.test_recovery_campaign_plan import study


def test_schedule_requires_entire_ordered_plan():
    plan=build(study())
    invocation.validate_schedule(plan,[[0,129],[129,2436]])
    for spans in ([[1,2436]],[[0,128],[129,2436]],[[0,2435]],[[0,0],[0,2436]]):
        with pytest.raises(ValueError):invocation.validate_schedule(plan,spans)


def test_handoff_timing_frozen_declaration_and_last_flag(tmp_path,monkeypatch):
    config=study();root=tmp_path/'archive';calls=[]
    class Worker:
        def __init__(self,*args):pass
        def close(self):pass
    monkeypatch.setattr(invocation,'SupervisedWork',Worker)
    monkeypatch.setattr(invocation,'completed_prefix',lambda root,plan:0)
    def execute(root,study,index,worker,protocol,runtime,started_at,last):
        now=time.perf_counter()
        calls.append(dict(index=index,start=started_at,last=last,settled=now))
        return dict(settled_at=now)
    monkeypatch.setattr(invocation,'execute_launch_unit',execute)
    started=time.perf_counter()
    receipt=invocation.run_invocation(root,config,'protocol',{},[[0,2],[2,2436]],started)
    assert receipt['status']=='complete' and receipt['completed_after']==2
    assert calls[0]['start']==started and calls[1]['start']==calls[0]['settled']
    assert [c['last'] for c in calls]==[False,True]
    with pytest.raises(ValueError,match='declaration'):
        invocation.run_invocation(root,config,'changed',{},[[0,2],[2,2436]])


def test_admission_failure_receipt_and_terminal_reentry(tmp_path,monkeypatch):
    def fail(root,plan):raise ValueError('bad ledger')
    monkeypatch.setattr(invocation,'completed_prefix',fail)
    root=tmp_path/'archive'
    with pytest.raises(ValueError,match='bad ledger'):
        invocation.run_invocation(root,study(),'protocol',{},[[0,2436]])
    path=tmp_path/'archive_invocations'/'invocation_000001.json'
    receipt=json.loads(path.read_text())
    assert receipt['status']=='failed' and receipt['elapsed_seconds']>=0
    with pytest.raises(RuntimeError,match='terminal'):
        invocation.run_invocation(root,study(),'protocol',{},[[0,2436]])
