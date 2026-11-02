import copy
import numpy as np
import pytest
import torch
from src.algos.reference_baseline_mappo import ReferenceBaselineMAPPO
from src.algos.action_marginal_mappo import ActionMarginalMAPPO
from src.env.joint_env import JointRetailEnv


def test_action_independent_reference_score_identity_and_variance_example():
    # Exact enumeration, no empirical sampling or environment steps. Shared
    # exogenous noise dominates this toy; an unrelated noisy reference can hurt.
    p=.4;actions=np.array([0.,1.]);score=actions-p;prob=np.array([1-p,p])
    raw=[];paired=[];harmful=[];weights=[]
    for noise in (-10.,10.):
        for a,s,q in zip(actions,score,prob):
            raw.append(s*(noise+2*a));paired.append(s*2*a)
            harmful.append(s*(noise+2*a+100*noise));weights.append(q/2)
        assert abs(np.dot(prob,score*noise))<1e-12
    mean=lambda v:float(np.dot(weights,v))
    var=lambda v:mean((np.asarray(v)-mean(v))**2)
    assert np.isclose(mean(raw),mean(paired)) and np.isclose(mean(raw),2*p*(1-p))
    assert var(paired)<var(raw)<var(harmful)


def test_zero_reference_equivalence_nonmutation_joint_update_and_schema():
    torch.set_num_threads(1);env=JointRetailEnv(episode_length_days=28,pricing_scope='per_sku')
    covers=np.linspace(4,10,10);old=ActionMarginalMAPPO(env,covers,100)
    ordinary=ReferenceBaselineMAPPO(env,covers,100,reference_baseline=False)
    zero=ReferenceBaselineMAPPO(env,covers,100);paired=ReferenceBaselineMAPPO(env,covers,100)
    torch.manual_seed(71236);episode=old.collect(env,87)
    original=copy.deepcopy(episode);zero_episode=dict(episode,reference_rewards=[0.]*28)
    metrics=old.update([episode]);assert ordinary.update([episode])==metrics
    assert zero.update([zero_episode])==metrics
    assert all(torch.equal(v,zero.state_dict()[k]) and torch.equal(v,ordinary.state_dict()[k]) for k,v in old.state_dict().items())
    # A fixed nonzero, nonconstant reference is an estimator change only.
    reference=np.linspace(0,100,28).tolist();ep=dict(episode,reference_rewards=reference)
    before={r:getattr(paired,r).weight.detach().clone() for r in ('pricing','ordering')}
    paired.update([ep]);assert all(not torch.equal(before[r],getattr(paired,r).weight) for r in before)
    assert not torch.equal(paired.ordering.weight,zero.ordering.weight)
    assert ep['rewards']==original['rewards'] and ep['reference_rewards']==reference
    for a,b in zip(ep['trace'],original['trace']):
        assert a['reward']==b['reward'] and np.array_equal(a['actions'],b['actions'])
    checkpoint=copy.deepcopy(paired.checkpoint());restored=ReferenceBaselineMAPPO(env,covers,101);restored.restore(checkpoint)
    with pytest.raises(ValueError,match='reference-baseline'):ordinary.restore(checkpoint)
    with pytest.raises(ValueError,match='reference-baseline'):restored.restore(old.checkpoint())
    with pytest.raises(ValueError,match='finite'):paired.update([dict(episode,reference_rewards=[float('nan')]*28)])
    env.get_state=lambda:(_ for _ in ()).throw(AssertionError('Global execution access'))
    assert paired.collect(env,88,greedy=True,training=False)['rewards']==restored.collect(env,88,greedy=True,training=False)['rewards']
