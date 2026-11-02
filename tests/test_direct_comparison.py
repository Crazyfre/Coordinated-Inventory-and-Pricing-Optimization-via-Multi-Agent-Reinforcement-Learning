import numpy as np
import pytest
import torch
from scripts.compare_joint_structured import analyze,compose,SEEDS,ARMS
from tests.test_joint_structured_mappo import setup


def test_bootstrap_preserves_pairing_and_negative_gates():
    rows=[]
    for key in range(3180000,3180032):
        rows.append(dict(arm='baseline',seed=0,key=key,reward_day=float(key),fill=.7))
        for seed in SEEDS:
            for arm in ARMS:
                rows.append(dict(arm=arm,seed=seed,key=key,reward_day=float(key)-(1 if arm=='mappo' else 0),fill=.7))
    result=analyze(rows)
    assert all(not v for v in result['gates'].values())
    for ref in ('baseline','ippo','separate'):
        assert result['effects'][ref+':reward_day']['ci991667']==[-1.,-1.]
    with pytest.raises(ValueError,match='grid'):analyze(rows[:-1])


def test_composition_uses_each_trained_actor_and_opposite_freeze():
    env,covers,p=setup('local');_,_,r=setup('local')
    frozen_before=r.ordering.weight.detach().clone()
    r.update([r.collect(env,k,frozen=('ordering',)) for k in (27,28)])
    assert torch.equal(frozen_before,r.ordering.weight)
    with torch.no_grad():p.pricing.weight.add_(.1);r.ordering.weight.add_(.2)
    combined=compose(p,r,env,covers,10)
    assert torch.equal(combined.pricing.weight,p.pricing.weight)
    assert torch.equal(combined.ordering.weight,r.ordering.weight)
