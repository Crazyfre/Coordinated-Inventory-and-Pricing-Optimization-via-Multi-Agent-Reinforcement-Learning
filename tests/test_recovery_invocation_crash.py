"""Real wrapper/ledger/owned-worker crashes; synthetic callback and tiny plan.

Scientific dispatcher/preflight is tested separately. No simulator runs here.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from scripts import recovery_invocation as invocation
from scripts.check_recovery_invocations import check
from scripts.check_recovery_attempts import check as check_attempts
from src.experiments.recovery_execution import execute_attempt

PLAN=dict(identity='wrapper-crash-fixture',units=[dict(item=str(i),phase='fit',days=2,
    seconds=20.,input_identity=str(i)) for i in range(2)],
    caps=dict(days=20,fit=100.,assessment=1.,total=101.),max_attempts=3)
DECLARATION=dict(study={'fixture':True},protocol='fixture',runtime={},schedule=[[0,2]])


def run(root,crash=None):
    original_build=invocation.build;original_execute=invocation.execute_launch_unit
    original_write=invocation.atomic_json
    def write(path,value):
        original_write(path,value)
        if crash=='before_boundaries' and value.get('status')=='running' and 'span' not in value:
            os._exit(23)
    invocation.atomic_json=write
    invocation.build=lambda study:PLAN
    def execute(root,study,index,worker,protocol,runtime,started_at,last):
        def boundary(event):
            if (index==0 and event==crash) or (index==1 and event=='committed' and crash=='final_committed'):
                os._exit(23)
        return execute_attempt(root,PLAN['identity'],PLAN['units'],PLAN['caps'],str(index),str(index),
            lambda budget:worker.call('tests.recovery_worker_callbacks','deterministic',{'days':2},budget),
            lambda:None,started_at=started_at,boundary=boundary,finalize=worker.close if last else lambda:None)
    invocation.execute_launch_unit=execute
    try:return invocation.run_invocation(root,**DECLARATION)
    finally:
        invocation.build=original_build;invocation.execute_launch_unit=original_execute
        invocation.atomic_json=original_write


@pytest.mark.skipif(os.name!='nt',reason='Windows owned-worker tests')
@pytest.mark.parametrize('point',['reserved','artifact_written','committed','final_committed'])
def test_actual_parent_exit_recovery_and_independent_receipts(tmp_path,point):
    root=tmp_path/'interrupted'
    code='from tests.test_recovery_invocation_crash import run; import sys; run(sys.argv[1],sys.argv[2])'
    result=subprocess.run([sys.executable,'-c',code,str(root),point],capture_output=True,text=True,
                          timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode==23,result.stderr
    receipt=run(root)
    assert receipt['status']=='complete' and receipt['completed_after']==2
    report=check(root,DECLARATION,PLAN)
    assert report['unknown_elapsed_invocations']==1
    attempts=check_attempts(root,PLAN,require_complete=True)
    assert attempts['charged']['days']==(4 if point in ('committed','final_committed') else 6)
    if point in ('committed','final_committed'):
        assert receipt['completed_before']==(2 if point=='final_committed' else 1) and attempts['attempts']==2
        if point=='final_committed':assert receipt['span']==[2,2]
    else:
        assert receipt['completed_before']==0 and attempts['by_status']['abandoned']['days']==2
    artifacts=[]
    for path in sorted(root.glob('attempt_*.json')):
        artifacts.append(json.loads(path.read_text())['payload'])
    assert all(payload=={'values':[0,1]} for payload in artifacts)


@pytest.mark.skipif(os.name!='nt',reason='Windows subprocess test')
def test_preboundary_crash_stops_before_any_scientific_attempt(tmp_path):
    root=tmp_path/'interrupted'
    code='from tests.test_recovery_invocation_crash import run; import sys; run(sys.argv[1],"before_boundaries")'
    result=subprocess.run([sys.executable,'-c',code,str(root)],capture_output=True,text=True,
                          timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode==23,result.stderr
    assert not (root/'attempts.sqlite').exists()
    with pytest.raises(RuntimeError,match='durable boundaries'):run(root)
    assert not (root/'attempts.sqlite').exists()
    receipt=json.loads((tmp_path/'interrupted_invocations'/'invocation_000001.json').read_text())
    assert receipt['status']=='failed' and receipt['elapsed_seconds'] is None
    with pytest.raises(RuntimeError,match='terminal'):run(root)
