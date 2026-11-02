import subprocess
import sys
import pytest
from src.experiments.recovery_ledger import RecoveryLedger

UNITS=[dict(item='a',phase='fit',days=10,seconds=5.,input_identity='input-a'),
       dict(item='b',phase='assessment',days=2,seconds=2.,input_identity='input-b')]
CAPS=dict(days=40,fit=20.,assessment=5.,total=25.)


def test_interrupted_attempt_retains_full_cost_and_same_inputs(tmp_path):
    path=tmp_path/'ledger.db';r=RecoveryLedger(path,'id',UNITS,CAPS)
    r.begin('a','input-a');r.close()
    r=RecoveryLedger(path,'id',UNITS,CAPS)
    with pytest.raises(RuntimeError):r.begin('a','input-a')
    r.recover()
    assert r.totals()==dict(days=10,fit=5.,assessment=0.,total=5.)
    with pytest.raises(ValueError):r.begin('a','new-input')
    with pytest.raises(ValueError):r.begin('b','input-b')
    n=r.begin('a','input-a');r.finish(n,8,3.,'attempt2.json','a'*64)
    assert r.totals()==dict(days=18,fit=8.,assessment=0.,total=8.)
    n=r.begin('b','input-b');r.finish(n,2,1.,'attempt3.json','b'*64)
    assert [x['status'] for x in r.attempts()]==['abandoned','complete','complete']
    with pytest.raises(ValueError):r.begin('b','input-b')
    r.close()


def test_limits_and_invalid_completion_are_terminal(tmp_path):
    r=RecoveryLedger(tmp_path/'retries.db','id',UNITS,CAPS)
    for i in range(3):r.begin('a','input-a');r.recover()
    with pytest.raises(RuntimeError,match='attempt limit'):r.begin('a','input-a')
    r.close()
    caps=dict(CAPS,days=15);r=RecoveryLedger(tmp_path/'cap.db','id',UNITS,caps)
    r.begin('a','input-a');r.recover()
    with pytest.raises(RuntimeError,match='budget'):r.begin('a','input-a')
    r.close()
    r=RecoveryLedger(tmp_path/'failed.db','id',UNITS,CAPS)
    n=r.begin('a','input-a')
    with pytest.raises(ValueError):r.finish(n,11,1.,'bad','a'*64)
    assert r.attempts()[0]['status']=='failed' and r.totals()['days']==10
    with pytest.raises(RuntimeError):r.recover()
    r.close()


def test_hard_exit_after_reservation_and_plan_guard(tmp_path):
    path=tmp_path/'crash.db'
    code="""
import os,sys,json
from src.experiments.recovery_ledger import RecoveryLedger
r=RecoveryLedger(sys.argv[1],'id',json.loads(sys.argv[2]),json.loads(sys.argv[3]))
r.begin('a','input-a')
os._exit(23)
"""
    import json
    result=subprocess.run([sys.executable,'-c',code,str(path),json.dumps(UNITS),json.dumps(CAPS)],
                          capture_output=True,text=True,timeout=30)
    assert result.returncode==23,result.stderr
    with pytest.raises(ValueError):RecoveryLedger(path,'changed',UNITS,CAPS)
    r=RecoveryLedger(path,'id',UNITS,CAPS);r.recover()
    assert r.attempts()[0]['status']=='abandoned'
    assert r.totals()['days']==10 and r.totals()['total']==5
    n=r.begin('a','input-a');r.finish(n,10,4.,'replayed.json','c'*64);r.close()
