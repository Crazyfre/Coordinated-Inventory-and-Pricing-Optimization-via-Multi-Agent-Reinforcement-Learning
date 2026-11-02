import json
import pytest
from src.experiments.recovery_execution import execute_attempt
from src.experiments.recovery_ledger import RecoveryLedger


def test_cleanup_time_is_charged_before_success_commit(tmp_path):
    now=[10.]
    units=[dict(item='a',phase='fit',days=0,seconds=20.,input_identity='a')]
    caps=dict(days=1,fit=40.,assessment=1.,total=41.)
    calls=[]
    def cleanup():
        calls.append(True);now[0]+=7.
    result=execute_attempt(tmp_path/'archive','test',units,caps,'a','a',lambda b:{},lambda:None,
                           clock=lambda:now[0],finalize=cleanup)
    assert result['charged']['fit']==7. and calls==[True]


def test_cleanup_overrun_and_work_failure_remain_terminal(tmp_path):
    for failure in (False,True):
        now=[10.];calls=[]
        units=[dict(item='a',phase='fit',days=0,seconds=5.,input_identity='a')]
        caps=dict(days=1,fit=40.,assessment=1.,total=41.)
        root=tmp_path/str(failure)
        def work(b):
            if failure:raise ValueError('work failed')
            return {}
        def cleanup():calls.append(True);now[0]+=7.
        with pytest.raises((ValueError,TimeoutError)):
            execute_attempt(root,'test',units,caps,'a','a',work,lambda:None,
                            clock=lambda:now[0],finalize=cleanup)
        assert calls==[True]
        receipt=json.loads((root/'failure_000001.json').read_text())
        assert receipt['observed_seconds']==7. and receipt['observed_overrun_seconds']==2.
        ledger=RecoveryLedger(root/'attempts.sqlite','test',units,caps)
        assert ledger.attempts()[0]['status']=='failed';ledger.close()
