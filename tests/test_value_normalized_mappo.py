import copy
import numpy as np
import pytest
import torch
from src.algos.adaptive_structured_mappo import AdaptiveStructuredMAPPO
from src.algos.value_normalized_mappo import RunningValueNorm,ValueNormalizedMAPPO
from src.algos.parameter_policy_gradient import order_action
from src.env.joint_env import JointRetailEnv


def test_value_moments_equal_role_weight_inverse_and_constant_floor():
    norm=RunningValueNorm()
    x=torch.tensor([-7.,0.,19.])
    assert torch.equal(norm.denormalize(x),x)
    norm.update([torch.tensor([1.,3.]),torch.tensor([5.]*28)])
    mean,std=norm.moments()
    assert float(mean)==pytest.approx(3.5)
    assert float(std)==pytest.approx(np.sqrt(2.75))
    assert torch.allclose(norm.denormalize(norm.normalize(x)),x,atol=2e-6,rtol=0)
    other=RunningValueNorm();other.update([torch.ones(4)*8,torch.ones(28)*8])
    assert float(other.moments()[1])==pytest.approx(.1)
    with pytest.raises(ValueError):norm.update([torch.tensor([float('nan')]),torch.ones(3)])


def test_normalized_values_first_actor_equivalence_roundtrip_and_local_execution():
    torch.set_num_threads(1)
    env=JointRetailEnv(episode_length_days=28,pricing_scope='per_sku');covers=np.linspace(4,10,10)
    model=ValueNormalizedMAPPO(env,covers,93);raw=AdaptiveStructuredMAPPO(env,covers,93)
    batch=[model.collect(env,k) for k in (21,22,23,24)]
    # First rollout values are zero in both arms; critic normalization must not
    # alter this first actor gradient or leak normalized units into GAE.
    raw.update(copy.deepcopy(batch));model.update(batch)
    for role in ('pricing','ordering'):
        assert all(torch.equal(a,b) for a,b in zip(getattr(raw,role).parameters(),getattr(model,role).parameters()))
    assert int(model.value_norm.updates)==1 and float(model.value_norm.moments()[1])>0
    state=batch[0]['rows']['ordering'][0]['state']
    with torch.no_grad():
        expected=model.value_norm.denormalize(model.critic('ordering')(torch.as_tensor(state)).squeeze())
    assert model.value('ordering',state)==float(expected)
    pricing=model.pricing.weight.detach().clone();ordering=model.ordering.weight.detach().clone()
    model.update([model.collect(env,k) for k in (31,32,33,34)])
    assert not torch.equal(pricing,model.pricing.weight) and not torch.equal(ordering,model.ordering.weight)
    checkpoint=copy.deepcopy(model.checkpoint());restored=ValueNormalizedMAPPO(env,covers,99);restored.restore(checkpoint)
    assert restored.value('ordering',state)==model.value('ordering',state)
    env.get_state=lambda:(_ for _ in ()).throw(AssertionError('global execution access'))
    assert restored.collect(env,41,greedy=True,training=False)['rewards']==model.collect(env,41,greedy=True,training=False)['rewards']
    fresh=ValueNormalizedMAPPO(env,covers,94,critic_mode='local')
    for row in fresh.collect(env,42,greedy=True,training=False)['trace']:
        assert np.array_equal(row['actions'],order_action(row['obs'],row['day'],covers))
    with pytest.raises(ValueError,match='normalized'):restored.restore(raw.checkpoint())
    with pytest.raises(ValueError,match='adaptive'):raw.restore(checkpoint)
