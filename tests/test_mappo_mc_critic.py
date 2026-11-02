import copy
import numpy as np
import pytest
import torch
from src.algos.mappo import interval_gae
from src.algos.structured_mappo import StructuredMAPPO
from src.env.joint_env import JointRetailEnv


def test_mc_critic_uses_complete_returns_without_changing_first_actor_update():
    torch.set_num_threads(1)
    env=JointRetailEnv(episode_length_days=28,pricing_scope='per_sku')
    covers=np.full(10,7.5)
    gae=StructuredMAPPO(env,covers,47)
    mc=StructuredMAPPO(env,covers,47,critic_target='mc')
    batch=[gae.collect(env,k) for k in (51,52)]
    result_gae=gae.update(copy.deepcopy(batch));result_mc=mc.update(copy.deepcopy(batch))
    for role in ('pricing','ordering'):
        assert torch.equal(getattr(gae,role).weight,getattr(mc,role).weight)
        expected=[]
        for ep in batch:
            rows=ep['rows'][role]
            # With lambda1 and terminal bootstrap0, GAE returns equal full MC
            # even with a nonzero baseline. This independently checks targets.
            _,returns=interval_gae(np.array(ep['rewards'])/100,[r['value'] for r in rows],
                [r['start'] for r in rows],gae.gamma,1.)
            expected.extend(returns)
        assert np.isclose(result_mc[role+'_critic_target_mean'],np.mean(expected),atol=2e-6)
        assert result_mc[role+'_critic_target_mean']!=result_gae[role+'_critic_target_mean']
    with pytest.raises(ValueError,match='critic target'):gae.restore(mc.checkpoint())
    other=StructuredMAPPO(env,covers,90,critic_target='mc');other.restore(mc.checkpoint())
    assert all(torch.equal(v,other.state_dict()[k]) for k,v in mc.state_dict().items())
