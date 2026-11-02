import json
import subprocess
import sys
import pytest
from src.experiments.recovery_execution import execute_attempt
from src.experiments.recovery_ledger import RecoveryLedger
from src.experiments.execution_guard import RunLock

UNITS=[dict(item='a',phase='fit',days=2,seconds=20.,input_identity='fixed')]
CAPS=dict(days=10,fit=100.,assessment=10.,total=110.)


def work(budget):
    for _ in range(2):budget.before_step()
    return dict(values=[1,4])


def run(root,**kwargs):
    return execute_attempt(root,'study',UNITS,CAPS,'a','fixed',work,lambda:None,**kwargs)


@pytest.mark.parametrize('point',['reserved','artifact_written','committed'])
def test_real_crash_boundaries_preserve_output_and_charges(tmp_path,point):
    clean=run(tmp_path/'clean');root=tmp_path/'crashed'
    code="""
import os,sys,json
from src.experiments.recovery_execution import execute_attempt
def work(b):
    b.before_step();b.before_step();return dict(values=[1,4])
def boundary(event):
    if event==sys.argv[2]:os._exit(23)
execute_attempt(sys.argv[1],'study',json.loads(sys.argv[3]),json.loads(sys.argv[4]),
    'a','fixed',work,lambda:None,boundary=boundary)
"""
    result=subprocess.run([sys.executable,'-c',code,str(root),point,json.dumps(UNITS),json.dumps(CAPS)],
                          capture_output=True,text=True,timeout=30)
    assert result.returncode==23,result.stderr
    if point=='committed':
        with pytest.raises(ValueError,match='all units'):run(root)
        artifact=root/'attempt_000001.json'
    else:
        recovered=run(root);artifact=root/'attempt_000002.json'
        assert recovered['charged']['days']==4
        assert recovered['charged']['fit']>=20.
    assert artifact.read_bytes()==__import__('pathlib').Path(clean['artifact']).read_bytes()
    ledger=RecoveryLedger(root/'attempts.sqlite','study',UNITS,CAPS)
    assert [r['status'] for r in ledger.attempts()]==(['complete'] if point=='committed' else ['abandoned','complete'])
    ledger.close()


def test_ownership_corruption_and_old_archive_guards(tmp_path):
    root=tmp_path/'owned'
    with RunLock(tmp_path/'owned.lock'):
        with pytest.raises(RuntimeError,match='owned'):run(root)
    units=UNITS+[dict(item='b',phase='fit',days=2,seconds=20.,input_identity='second')]
    result=execute_attempt(root,'study',units,CAPS,'a','fixed',work,lambda:None)
    from pathlib import Path
    Path(result['artifact']).write_text('changed')
    with pytest.raises(ValueError,match='corruption'):
        execute_attempt(root,'study',units,CAPS,'b','second',work,lambda:None)
    ledger=RecoveryLedger(root/'attempts.sqlite','study',units,CAPS)
    assert [r['status'] for r in ledger.attempts()]==['complete','failed']
    assert ledger.totals()['days']==4
    ledger.close()
    assert (root/'failure_000002.json').is_file()
    old=tmp_path/'old';old.mkdir()
    with pytest.raises(ValueError,match='migration'):run(old)


def test_step_overrun_and_input_failure_cannot_retry(tmp_path):
    def excessive(b):
        for _ in range(3):b.before_step()
    def invalid():raise ValueError('changed source')
    for name,callback,validate in [('steps',excessive,lambda:None),('inputs',work,invalid)]:
        root=tmp_path/name
        with pytest.raises((ValueError,RuntimeError)):
            execute_attempt(root,'study',UNITS,CAPS,'a','fixed',callback,validate)
        with pytest.raises(RuntimeError,match='terminal'):run(root)
        ledger=RecoveryLedger(root/'attempts.sqlite','study',UNITS,CAPS)
        assert ledger.totals()['days']==2 and ledger.totals()['fit']==20.
        ledger.close()
        receipt=json.loads((root/'failure_000001.json').read_text())
        assert receipt['stage']==('work' if name=='steps' else 'input_validation')
        assert receipt['parent_step_admissions']==(2 if name=='steps' else 0)
        assert ('simulator day reservation exhausted' if name=='steps' else 'changed source') in receipt['traceback']
