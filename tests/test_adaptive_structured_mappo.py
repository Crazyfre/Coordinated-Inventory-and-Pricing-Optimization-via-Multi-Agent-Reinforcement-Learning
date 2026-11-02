import copy
import numpy as np
import pytest
import torch
from src.algos.adaptive_structured_mappo import AdaptiveOrderingActor,AdaptiveStructuredMAPPO
from src.algos.structured_ordering_ppo import StructuredOrderingActor,ordering_features,execute_cover
from src.algos.parameter_policy_gradient import order_action
from src.algos.structured_mappo import StructuredMAPPO
from src.env.joint_env import JointRetailEnv


def test_initial_density_matches_fixed_policy_and_dispersion_has_score_gradient():
    covers=np.full(10,7.5,dtype=np.float32)
    fixed=StructuredOrderingActor(covers);adaptive=AdaptiveOrderingActor(covers)
    features=torch.ones((4,10,7));latent=torch.tensor(covers).repeat(4,1)+.25
    fixed_lp,_=fixed.stats(features,latent);adaptive_lp,_=adaptive.stats(features,latent)
    assert torch.allclose(fixed_lp,adaptive_lp,rtol=0,atol=1e-6)
    (-adaptive_lp.mean()).backward()
    assert torch.isfinite(adaptive.log_std.grad).all() and torch.all(adaptive.log_std.grad!=0)
    with torch.no_grad():adaptive.log_std.copy_(torch.linspace(-100,100,10))
    sd=adaptive.distribution(features).scale
    assert float(sd.min().detach())==pytest.approx(np.exp(-3)) and float(sd.max().detach())==pytest.approx(np.exp(.5))


def test_adaptive_joint_update_checkpoint_and_exact_greedy_anchor():
    torch.set_num_threads(1);env=JointRetailEnv(episode_length_days=28,pricing_scope='per_sku')
    covers=np.linspace(4,10,10,dtype=np.float32);model=AdaptiveStructuredMAPPO(env,covers,91)
    initial_sd=model.ordering.log_std.detach().clone()
    greedy=model.collect(env,3,greedy=True,training=False)
    for row in greedy['trace']:
        assert np.array_equal(row['actions'],order_action(row['obs'],row['day'],covers))
        if row['price_actions'] is not None:assert np.all(row['price_actions']==4)
    model.update([model.collect(env,k) for k in (7,8,9,10)])
    assert not torch.equal(initial_sd,model.ordering.log_std)
    checkpoint=copy.deepcopy(model.checkpoint())
    other=AdaptiveStructuredMAPPO(env,covers,92);other.restore(checkpoint)
    assert model.collect(env,13,greedy=True,training=False)['rewards']==other.collect(env,13,greedy=True,training=False)['rewards']
    fixed=StructuredMAPPO(env,covers)
    with pytest.raises(ValueError,match='architecture'):fixed.restore(checkpoint)
    with pytest.raises(ValueError,match='adaptive'):model.restore(fixed.checkpoint())
