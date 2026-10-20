import numpy as np
import torch
from src.experiments.phase4 import make_env, evaluate


def test_joint_ppo_inactive_price_head_has_zero_gradient():
    from src.algos.joint_ppo import JointPPO
    torch.set_num_threads(1)
    agent = JointPPO(make_env(days=9))
    x = torch.tensor(np.stack([agent.observation(),agent.observation()]),dtype=torch.float32)
    actions = torch.zeros((2,11),dtype=torch.long)
    lp, entropy, _ = agent.stats(x, actions, torch.zeros(2))
    (lp.sum()+entropy.sum()).backward()
    assert torch.count_nonzero(agent.net.logits_heads[0].weight.grad) == 0
    assert torch.count_nonzero(agent.net.logits_heads[1].weight.grad) > 0


def test_joint_ppo_trains_daily_with_weekly_mask_and_restores(tmp_path):
    from src.algos.joint_ppo import JointPPO
    torch.manual_seed(8)
    env = make_env(days=9)
    agent = JointPPO(env)
    history = agent.train(2, seed=8)
    assert history[-1]['daily_steps'] == 9
    assert history[-1]['pricing_decisions'] == 2
    assert all(np.isfinite(v) for v in history[-1].values())
    path = tmp_path/'joint.pt'
    agent.save(path)
    before = evaluate(env,agent.order_action,2,123,agent.pricing_action)
    restored = JointPPO(env)
    restored.load(path)
    after = evaluate(env,restored.order_action,2,123,restored.pricing_action)
    assert before == after


def test_joint_ppo_reports_post_update_kl_with_single_epoch():
    from src.algos.joint_ppo import JointPPO
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = JointPPO(make_env(days=14),epochs=1)
    assert model.train(1,seed=3)[0]['kl'] > 0
