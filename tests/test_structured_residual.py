import numpy as np
import torch
import pytest
from src.algos.structured_residual import StructuredResidualActor,StructuredResidualMAPPO
from src.algos.structured_imitation import InventoryPositionLinearPolicy
from src.env.joint_env import JointRetailEnv


def test_center_support_duplicates_logprob_and_gradients():
    torch.set_num_threads(1)
    actor=StructuredResidualActor(5,1,2,[6,-1,0],8)
    obs=torch.tensor([[0.,0,0,0,0],[0.,0,0,0,100],[10.,15,5,0,10.]])
    expected=InventoryPositionLinearPolicy(1,2,[6,-1,0]).predict(obs)
    np.testing.assert_array_equal(actor.centers(obs),expected)
    dist=actor(obs)[0]
    for i,support in enumerate([[0,1,2],[38,39,40],[4,5,6,7,8]]):
        assert torch.where(dist.probs[i]>0)[0].tolist()==support
    assert torch.all(dist.probs.gather(1,actor.centers(obs))>.998)
    actions=dist.sample().unsqueeze(-1)
    lp,entropy=actor.stats(obs,actions)
    assert torch.isfinite(lp).all() and torch.isfinite(entropy).all()
    (-lp.mean()-.01*entropy.mean()).backward()
    assert actor.heads[0].bias.grad.abs().sum()>0


def test_matching_checkpoint_roundtrip_and_vanilla_rejection(tmp_path):
    from src.algos.mappo import HeterogeneousMAPPO
    env=JointRetailEnv(episode_length_days=2)
    center=tmp_path/'center.json'
    InventoryPositionLinearPolicy(env.n,env.obs_pipe_horizon,[6,-1,0]).save(center)
    model=StructuredResidualMAPPO(env,center);path=tmp_path/'actor.pt';model.save(path)
    restored=StructuredResidualMAPPO(env,center);restored.load(path)
    obs,_=env.reset(seed=2)
    np.testing.assert_array_equal(model.replenishment.act(obs['replenishment'],True)[0],
                                  restored.replenishment.act(obs['replenishment'],True)[0])
    with pytest.raises(ValueError,match='variant'):
        HeterogeneousMAPPO(env).load(path)
    history=model.train(1,seed=3)
    assert np.isfinite(list(history[0].values())).all()
