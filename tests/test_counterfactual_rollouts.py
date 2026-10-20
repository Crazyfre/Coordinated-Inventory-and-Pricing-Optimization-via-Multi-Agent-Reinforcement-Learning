import copy
from types import SimpleNamespace
import pytest
from scripts.diagnose_counterfactual_rollouts import clone_environment, check_branches


def test_clone_detaches_mutable_state_and_discards_closure():
    env=SimpleNamespace(state=[1], pricing_hook=lambda day: 1)
    branch=clone_environment(env)
    branch.state.append(2)
    assert env.state==[1]
    assert branch.pricing_hook is None
    assert env.pricing_hook(0)==1


def test_pairing_detects_changed_noise_and_immediate_credit():
    ref=[dict(actions=[1,0], leads=[3,0], demand=[4,5], reward=2)]
    check_branches([dict(trace=copy.deepcopy(ref))],ref)
    for key,value in [('demand',[5,5]),('reward',3),('leads',[4,0])]:
        rows=copy.deepcopy(ref)
        rows[0][key]=value
        with pytest.raises(AssertionError):
            check_branches([dict(trace=rows)],ref)


def test_zero_order_does_not_require_stale_lead_match():
    ref=[dict(actions=[0],leads=[4],demand=[2],reward=1)]
    other=[dict(actions=[1],leads=[3],demand=[2],reward=1)]
    check_branches([dict(trace=other)],ref)
