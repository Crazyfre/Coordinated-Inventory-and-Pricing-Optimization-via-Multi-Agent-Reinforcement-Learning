import json
import pytest
from src.experiments.segmented_execution import execute_one
from src.experiments.execution_journal import digest

CAPS=dict(fit=10.,assessment=5.,total=12.)


def run_unit(root,name,now,cost,days=2,fail=False):
    def work(path,deadline):
        now[0]+=cost
        path.write_text(json.dumps(dict(unit=name,value=sum(ord(c) for c in name))))
        if fail:raise RuntimeError('simulated abrupt worker exit')
        return days
    return execute_one(root,'fixed',['a','b','freeze'],4,CAPS,name,
                       'assessment' if name=='freeze' else 'fit',days,work,clock=lambda:now[0])


def test_clean_boundaries_match_uninterrupted_artifacts_and_zero_day_time(tmp_path):
    now=[0.];a=tmp_path/'uninterrupted'
    for name in ['a','b','freeze']:run_unit(a,name,now,1,0 if name=='freeze' else 2)
    now=[100.];b=tmp_path/'segmented'
    for name in ['a','b','freeze']:
        run_unit(b,name,now,1,0 if name=='freeze' else 2);now[0]+=500 # offline gaps
    for i in range(3):assert digest(a/f'unit_{i:04d}.json')==digest(b/f'unit_{i:04d}.json')
    timer=json.loads((b/'time.json').read_text());assert timer['used']==dict(fit=2.,assessment=1.,total=3.)
    assert json.loads((b/'journal.json').read_text())['reserved_days']==4


def test_interruption_cannot_duplicate_or_skip_unit(tmp_path):
    root=tmp_path/'run';now=[0.]
    run_unit(root,'a',now,1)
    with pytest.raises(RuntimeError,match='simulated'):run_unit(root,'b',now,1,fail=True)
    for name in ['b','freeze']:
        with pytest.raises(RuntimeError):run_unit(root,name,now,1,0 if name=='freeze' else 2)
    assert json.loads((root/'journal.json').read_text())['pending']['item']=='b'
    assert json.loads((root/'time.json').read_text())['status']=='failed'


def test_startup_and_artifact_commit_crash_are_not_free(tmp_path):
    root=tmp_path/'startup';now=[11.];called=[]
    with pytest.raises(TimeoutError):
        execute_one(root,'fixed',['a'],2,CAPS,'a','fit',2,lambda *a:called.append(1),
                    clock=lambda:now[0],started_at=0.)
    assert not called and json.loads((root/'time.json').read_text())['used']['fit']==11
    # Crash boundary: journal committed, clock still marked running.
    root=tmp_path/'mismatch';now=[0.];run_unit(root,'a',now,1)
    path=root/'time.json';saved=json.loads(path.read_text());saved['completed']=[]
    path.write_text(json.dumps(saved))
    with pytest.raises(RuntimeError,match='mismatch'):run_unit(root,'b',now,1)


def test_process_exit_after_reservation_is_terminal(tmp_path):
    import subprocess
    import sys
    root=tmp_path/'abrupt'
    code="""
import os,sys
from src.experiments.segmented_execution import execute_one
def stop(path,deadline):
    os._exit(23)
execute_one(sys.argv[1],'fixed',['a','b','freeze'],4,
            dict(fit=10.,assessment=5.,total=12.),'a','fit',2,stop)
"""
    child=subprocess.run([sys.executable,'-c',code,str(root)],capture_output=True,text=True,timeout=30)
    assert child.returncode==23,child.stderr
    assert json.loads((root/'journal.json').read_text())['pending']==dict(item='a',days=2)
    assert json.loads((root/'time.json').read_text())['status']=='running'
    with pytest.raises(RuntimeError,match='interrupted'):run_unit(root,'a',[0.],1)
