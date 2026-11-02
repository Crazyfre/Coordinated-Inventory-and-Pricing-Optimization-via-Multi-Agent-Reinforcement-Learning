import pytest
from scripts.recovery_overhead import combined_admission
from scripts.recovery_invocation import launch
from src.experiments.recovery_ledger import RecoveryLedger


def test_attempt_alone_fits_but_combined_reservation_rejected(tmp_path):
    plan=dict(identity='fixture',units=[dict(item=str(i),phase='fit',days=1,seconds=4.,input_identity=str(i)) for i in range(2)],
              caps=dict(days=10,fit=10.,assessment=10.,total=20.),max_attempts=3)
    root=tmp_path/'archive';root.mkdir()
    ledger=RecoveryLedger(root/'attempts.sqlite',**plan)
    attempt=ledger.begin('0','0');ledger.finish(attempt,1,4.,'artifact','0'*64);ledger.close()
    assert combined_admission(root,plan,dict(fit=2.,assessment=0.,total=2.),1)['fit']==10.
    with pytest.raises(RuntimeError,match='combined'):
        combined_admission(root,plan,dict(fit=3.,assessment=0.,total=3.),1)
    assert combined_admission(root,plan,dict(fit=3.,assessment=0.,total=3.))['fit']==7.


def test_scientific_launch_requires_explicit_overhead(tmp_path):
    with pytest.raises(ValueError,match='overhead'):
        launch(tmp_path/'archive',dict(study={},protocol='fixture',runtime={},schedule=[]))
    assert not (tmp_path/'archive').exists()
