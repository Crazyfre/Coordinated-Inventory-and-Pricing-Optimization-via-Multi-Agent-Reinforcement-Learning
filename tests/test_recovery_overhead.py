import pytest
from scripts.recovery_overhead import charges

POLICY=dict(per_invocation_seconds=2.,maximum_invocations=3,fit=4.,assessment=4.,total=6.)
PLAN=dict(units=[dict(phase='fit'),dict(phase='assessment')])


def test_interruption_reservation_and_completed_tail():
    records=[dict(status='interrupted',span=[0,1]),dict(status='complete',span=[0,1],trailing_seconds=.25),
             dict(status='running',span=[1,2])]
    assert charges(records,POLICY,PLAN)==dict(fit=2.25,assessment=2.,total=4.25)


@pytest.mark.parametrize('records',[
    [dict(status='complete',span=[0,1],trailing_seconds=2.1)],
    [dict(status='running',span=[0,1])]*3,
    [dict(status='running',span=[1,2])]*4,
    [dict(status='failed',span=[0,1])],
])
def test_caps_and_failure_reject_continuation(records):
    with pytest.raises(RuntimeError):charges(records,POLICY,PLAN)


def test_wrapper_overhead_reservation_precedes_work(tmp_path,monkeypatch):
    from scripts import recovery_invocation as wrapper
    from tests.test_recovery_campaign_plan import study
    calls=[]
    monkeypatch.setattr(wrapper,'execute_launch_unit',lambda *a,**k:calls.append(True))
    policy=dict(POLICY,per_invocation_seconds=5.)
    with pytest.raises(RuntimeError,match='overhead budget'):
        wrapper.run_invocation(tmp_path/'archive',study(),'fixture',{},[[0,2436]],overhead=policy)
    assert calls==[]
