import numpy as np
import torch
import pytest
from src.experiments.phase4 import make_env


def test_signed_log_transform_preserves_zero_sign_and_bounds_large_inputs():
    from src.algos.stability import SignedLog
    x = torch.tensor([-1000., 0., 1000.])
    y = SignedLog()(x)
    assert torch.allclose(y, torch.sign(x)*torch.log1p(torch.abs(x)))
    assert y[0] == -y[2]
    assert y.abs().max() < 7


def test_signed_log_handles_unbounded_capacity_without_nan():
    from src.algos.stability import SignedLog
    y = SignedLog()(torch.tensor([float('inf'),float('-inf')]))
    assert torch.isfinite(y).all()
    assert y[0] > 50
    assert y[0] == -y[1]


def test_r3_scaled_normalized_training_is_finite_and_checkpoint_restores(tmp_path):
    from src.agents.replenishment_agent import ReplenishmentPPOAgent
    torch.set_num_threads(1)
    torch.manual_seed(1)
    env = make_env(days=9, joint=False)
    agent = ReplenishmentPPOAgent(env, rollout_steps=18, normalize_inputs=True, reward_scale=100.)
    metrics = agent.train(2, seed=1)
    assert np.isfinite(metrics[-1]['v_loss'])
    assert metrics[-1]['v_loss'] < 1e4
    obs, _ = env.reset(seed=5)
    before = agent.act(obs, True)[0]
    path = tmp_path/'r3.pt'
    agent.save(path)
    fresh = ReplenishmentPPOAgent(env, normalize_inputs=True, reward_scale=100.)
    fresh.load(path)
    assert np.array_equal(before, fresh.act(obs, True)[0])


def test_r3_default_loader_detects_normalized_checkpoint(tmp_path):
    from src.agents.replenishment_agent import ReplenishmentPPOAgent
    torch.set_num_threads(1)
    env = make_env(days=9,joint=False)
    model = ReplenishmentPPOAgent(env,normalize_inputs=True,reward_scale=100.)
    path = tmp_path/'r3_normalized.pt'
    model.save(path)
    restored = ReplenishmentPPOAgent(env)
    restored.load(path)
    obs,_ = env.reset(seed=3)
    assert model.act(obs,True)[2] == restored.act(obs,True)[2]
    assert np.array_equal(model.act(obs,True)[0],restored.act(obs,True)[0])
    assert restored.reward_scale == 100.


def test_mappo_normalized_checkpoint_tracks_policy_diagnostics(tmp_path):
    from src.algos.mappo import HeterogeneousMAPPO
    torch.set_num_threads(1)
    torch.manual_seed(7)
    env = make_env(days=14)
    agent = HeterogeneousMAPPO(env, normalize_inputs=True, target_kl=.02)
    row = agent.train(2,seed=7)[-1]
    for name in ['pricing_kl', 'replenishment_kl', 'pricing_entropy', 'replenishment_entropy']:
        assert np.isfinite(row[name])
    path = tmp_path/'mappo.pt'
    agent.save(path)
    restored = HeterogeneousMAPPO(env, normalize_inputs=True)
    restored.load(path)
    obs,_ = env.reset(seed=4)
    assert np.array_equal(agent.pricing.act(obs['pricing'],True)[0], restored.pricing.act(obs['pricing'],True)[0])


def test_mappo_default_loader_detects_normalized_checkpoint_for_evaluation(tmp_path):
    from src.algos.mappo import HeterogeneousMAPPO
    torch.set_num_threads(1)
    env = make_env(days=9)
    model = HeterogeneousMAPPO(env,normalize_inputs=True)
    path = tmp_path/'normalized.pt'
    model.save(path)
    restored = HeterogeneousMAPPO(env)
    restored.load(path)
    obs,_ = env.reset(seed=7)
    state = env.get_state()
    assert model.value(state) == restored.value(state)
    assert np.array_equal(model.pricing.act(obs['pricing'],True)[0],restored.pricing.act(obs['pricing'],True)[0])


@pytest.mark.parametrize('method',['r3','mappo','joint','independent'])
def test_trainers_reject_nonfinite_reward_scale(method):
    from src.agents.replenishment_agent import ReplenishmentPPOAgent
    from src.algos.mappo import HeterogeneousMAPPO
    from src.algos.joint_ppo import JointPPO
    from src.algos.independent_ppo import IndependentPPO
    torch.set_num_threads(1)
    env = make_env(days=9,joint=method!='r3')
    cls = {'r3':ReplenishmentPPOAgent,'mappo':HeterogeneousMAPPO,
           'joint':JointPPO,'independent':IndependentPPO}[method]
    with pytest.raises(ValueError,match='reward_scale'):
        cls(env,reward_scale=float('nan'))


def test_mappo_reports_post_update_kl_with_single_epoch():
    from src.algos.mappo import HeterogeneousMAPPO
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = HeterogeneousMAPPO(make_env(days=14),epochs=1,normalize_inputs=True)
    row = model.train(1,seed=3)[0]
    assert row['pricing_kl'] > 0
    assert row['replenishment_kl'] > 0
