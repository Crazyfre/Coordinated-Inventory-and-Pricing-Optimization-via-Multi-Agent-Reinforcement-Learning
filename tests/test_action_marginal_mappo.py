import copy
import numpy as np
import pytest
import torch
from outputs.action_marginal_mappo import ActionMarginalOrderingActor,ActionMarginalMAPPO
from src.algos.value_normalized_mappo import ValueNormalizedMAPPO
from src.env.joint_env import JointRetailEnv


def test_grid_mass_sampling_threshold_gradient_and_degenerate_action():
    torch.set_num_threads(1);actor=ActionMarginalOrderingActor(np.full(10,7.5))
    f=torch.zeros(10,9);f[:,0]=1;f[:,7]=15;f[:,8]=2
    logmass=actor.grid_log_masses(f)
    assert torch.allclose(logmass.logsumexp(-1),torch.zeros(10),atol=1e-6,rtol=0)
    assert torch.allclose(logmass[:,0].exp(),torch.full((10,),.5),atol=1e-6,rtol=0)
    assert torch.isneginf(logmass[:,6:]).all()
    lp,_=actor.stats(f,torch.zeros(10));lp.backward()
    assert torch.allclose(actor.weight.grad[:,0],torch.full((10,),-np.sqrt(2/np.pi)/.75),atol=1e-5,rtol=0)
    assert torch.allclose(actor.log_std.grad,torch.zeros(10),atol=1e-5,rtol=0)
    actor.zero_grad();f[:,7]=100;f[:,8]=1
    lp,entropy=actor.stats(f,torch.zeros(10));assert float(lp.detach())==0 and float(entropy.detach())==0
    lp.backward();assert torch.count_nonzero(actor.weight.grad)==0 and torch.count_nonzero(actor.log_std.grad)==0
    # Include zero forecast, upper cover clipping and the maximum order bin.
    actor=ActionMarginalOrderingActor(np.linspace(.5,19.5,10));f[:,7]=torch.linspace(0,25,10)
    f[:,8]=torch.tensor([0,.1,.5,1,2,5,10,20,50,100.])
    mass=actor.grid_log_masses(f)
    assert torch.allclose(mass.logsumexp(-1),torch.zeros(10),atol=1e-6,rtol=0)
    torch.manual_seed(71993);z=actor.distribution(f).sample((20000,)).numpy().astype(np.float32)
    actual=np.ceil(np.maximum(z.clip(0,20)*f[:,8].numpy()-f[:,7].numpy(),0)/5).clip(0,40).astype(int)
    probabilities=actor.grid_distribution(f).probs.detach().numpy()
    for sku in range(10):
        empirical=np.bincount(actual[:,sku],minlength=41)/len(actual)
        assert np.max(np.abs(empirical-probabilities[sku]))<.025


def test_joint_physical_equivalence_update_checkpoint_and_local_execution():
    torch.set_num_threads(1);env=JointRetailEnv(episode_length_days=28,pricing_scope='per_sku')
    covers=np.linspace(4,10,10);raw=ValueNormalizedMAPPO(env,covers,96);marginal=ActionMarginalMAPPO(env,covers,96)
    assert all(torch.equal(v,marginal.state_dict()[k]) for k,v in raw.state_dict().items())
    torch.manual_seed(71234);a=raw.collect(env,83)
    torch.manual_seed(71234);b=marginal.collect(env,83)
    assert a['rewards']==b['rewards']
    for x,y,row in zip(a['trace'],b['trace'],b['rows']['ordering']):
        assert np.array_equal(x['actions'],y['actions']) and np.array_equal(x['price_actions'],y['price_actions'])
        assert row['features'].shape==(10,9) and np.array_equal(row['action'],y['actions'])
    before={r:getattr(marginal,r).weight.detach().clone() for r in ('pricing','ordering')}
    metrics=marginal.update([b]);assert np.isfinite(metrics['critic_loss'])
    assert all(not torch.equal(before[r],getattr(marginal,r).weight) for r in before)
    checkpoint=copy.deepcopy(marginal.checkpoint());restored=ActionMarginalMAPPO(env,covers,97);restored.restore(checkpoint)
    env.get_state=lambda:(_ for _ in ()).throw(AssertionError('Global execution access'))
    assert marginal.collect(env,85,greedy=True,training=False)['rewards']==restored.collect(env,85,greedy=True,training=False)['rewards']
    with pytest.raises(ValueError,match='action-marginal'):restored.restore(raw.checkpoint())
    with pytest.raises(ValueError,match='normalized'):raw.restore(checkpoint)
