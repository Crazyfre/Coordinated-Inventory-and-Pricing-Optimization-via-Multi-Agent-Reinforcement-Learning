from pathlib import Path
import pytest
from src.experiments import recovery_execution as execution
from src.experiments.recovery_ledger import RecoveryLedger


def test_explicit_dependency_hashing_and_full_reentry(tmp_path,monkeypatch):
    units=[dict(item=str(i),phase='fit',days=0,seconds=20.,input_identity=str(i)) for i in range(4)]
    caps=dict(days=1,fit=100.,assessment=1.,total=101.)
    calls=[]
    original=execution.digest
    def recording(path):
        calls.append(Path(path).name)
        return original(path)
    monkeypatch.setattr(execution,'digest',recording)
    def run(index,scope=None):
        return execution.execute_attempt(tmp_path/'archive','fixture',units,caps,str(index),str(index),
            lambda b:{'index':index},lambda:None,committed_items=scope)
    first=run(0);run(1)
    calls.clear()
    run(2,{'1'})
    assert 'attempt_000001.json' not in calls
    assert 'attempt_000002.json' in calls and 'attempt_000003.json' in calls
    Path(first['artifact']).write_text('corrupt unused artifact')
    # A fresh invocation uses full-prefix validation and detects the corruption.
    with pytest.raises(ValueError,match='corruption'):run(3)
    ledger=RecoveryLedger(tmp_path/'archive'/'attempts.sqlite','fixture',units,caps)
    assert ledger.attempts()[-1]['status']=='failed'
    ledger.close()


def test_missing_dependency_fails_before_callback(tmp_path):
    units=[dict(item='a',phase='fit',days=0,seconds=10.,input_identity='a')]
    called=[]
    with pytest.raises(ValueError,match='missing'):
        execution.execute_attempt(tmp_path/'archive','fixture',units,
            dict(days=1,fit=20.,assessment=1.,total=21.),'a','a',
            lambda b:called.append(True),lambda:None,committed_items={'missing'})
    assert not called
