import numpy as np
import torch
from src.experiments.phase4 import make_env, evaluate


def test_independent_ppo_uses_disjoint_local_critics_and_no_global_state(tmp_path):
    from src.algos.independent_ppo import IndependentPPO
    torch.set_num_threads(1)
    torch.manual_seed(5)
    env = make_env(days=9)
    def forbidden():
        raise AssertionError('independent PPO cannot use global state')
    env.get_state = forbidden
    agent = IndependentPPO(env)
    p_ids = {id(p) for p in agent.pricing.parameters()}
    r_ids = {id(p) for p in agent.replenishment.parameters()}
    assert p_ids.isdisjoint(r_ids)
    history = agent.train(2,seed=5)
    assert history[-1]['pricing_decisions'] == 2
    assert history[-1]['daily_steps'] == 9
    assert all(np.isfinite(x) for x in history[-1].values())
    path = tmp_path/'independent.pt'
    agent.save(path)
    before = evaluate(env,agent.order_action,2,99,agent.pricing_action)
    restored = IndependentPPO(env)
    restored.load(path)
    assert before == evaluate(env,restored.order_action,2,99,restored.pricing_action)


def test_independent_ppo_reports_post_update_kl_with_single_epoch():
    from src.algos.independent_ppo import IndependentPPO
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = IndependentPPO(make_env(days=14),epochs=1)
    row = model.train(1,seed=3)[0]
    assert row['pricing_kl'] > 0
    assert row['replenishment_kl'] > 0
