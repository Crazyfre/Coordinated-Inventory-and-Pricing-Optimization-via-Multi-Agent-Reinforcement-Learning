import numpy as np
import pytest
from src.algos.joint_parameter_policy import initial_parameters, price_action, ordering_action, JointParameterGradient
from src.algos.parameter_policy_gradient import order_action


def test_exact_strong_start_and_legal_price_rounding():
    rng=np.random.default_rng(9);covers=np.linspace(6,10,10);p=initial_parameters(covers)
    for day in [1,8,15,22]:
        x=rng.uniform(0,50,187).astype(np.float32)
        np.testing.assert_array_equal(price_action(x,day,p),np.full(10,4))
        y=rng.uniform(0,50,260).astype(np.float32)
        np.testing.assert_array_equal(ordering_action(y,day,p),order_action(y,day,covers))
    p[10:20]=np.linspace(-2,6,10)
    a=price_action(x,1,p)
    assert np.all((a>=0)&(a<=4)) and len(np.unique(a))>1


def test_inventory_time_inputs_and_role_freeze():
    p=initial_parameters(np.full(10,7.5));p[10:20]=2;p[20:30]=1;p[30]=1
    x=np.zeros(187,np.float32);x[-20:-10]=1
    y=x.copy();y[:10]=16
    assert np.any(price_action(x,1,p)!=price_action(y,1,p))
    for arm,frozen in [('ordering',slice(10,None)),('pricing',slice(0,10))]:
        learner=JointParameterGradient(np.full(10,7.5),arm,501);before=learner.mean.copy()
        e=learner.perturbations();learner.update(e,np.arange(8))
        np.testing.assert_array_equal(learner.mean[frozen],before[frozen])
        np.testing.assert_array_equal(e[:,frozen],0)
    with pytest.raises(ValueError):price_action(x,29,p)


def test_pilot_eligibility_requires_both_references_and_service():
    from scripts.pilot_joint_parameters import analyze
    labels=['joint','independent','ordering','pricing','start','full7_5']
    rows=[dict(label=l,scenario=k,reward_day=100.+(1 if l=='joint' else 0),fill=.65,price_counts=[0,0,0,0,40])
          for k in range(1710000,1710064) for l in labels]
    assert analyze(rows)['eligible_for_new_confirmation']
    for r in rows:
        if r['label']=='independent':r['reward_day']=100.8
    assert not analyze(rows)['eligible_for_new_confirmation']
    for r in rows:
        if r['label']=='independent':r['reward_day']=100.
        if r['label']=='joint':r['fill']=.64
    assert not analyze(rows)['eligible_for_new_confirmation']
