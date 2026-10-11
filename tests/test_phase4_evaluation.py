import numpy as np
from src.experiments.phase4 import make_env


def test_r3_diagnostics_describe_transition_day_not_next_arrivals():
    env = make_env(days=4, joint=False)
    env.reset(seed=3)
    _, _, _, _, info = env.step(np.zeros(env.n, dtype=np.int64))
    assert info['day'] == 1
    assert info['orders_delivered'].sum() == 0


def test_lead_scenarios_are_paired_by_day_and_sku():
    for joint in (False, True):
        env = make_env(days=9, joint=joint)
        env.reset(seed=3)
        all_orders = env.pipeline.place_orders(np.ones(env.n), day=1)
        env.reset(seed=3)
        only_last = np.zeros(env.n)
        only_last[-1] = 1
        sparse = env.pipeline.place_orders(only_last, day=1)
        assert sparse[-1] == all_orders[-1]
        env.reset(seed=3)
        day2 = env.pipeline.place_orders(only_last, day=2)
        env.reset(seed=3)
        env.pipeline.place_orders(np.ones(env.n), day=1)
        after_day1 = env.pipeline.place_orders(only_last, day=2)
        assert after_day1[-1] == day2[-1]


def test_r3_checkpoint_restores_actions(tmp_path):
    import torch
    from src.agents.replenishment_agent import ReplenishmentPPOAgent
    torch.set_num_threads(1)
    env = make_env(days=4, joint=False)
    agent = ReplenishmentPPOAgent(env, hidden=16, rollout_steps=4, epochs_per_update=1)
    agent.train(1, seed=3)
    obs, _ = env.reset(seed=9)
    expected = agent.act(obs, greedy=True)[0]
    path = tmp_path / 'r3.pt'
    agent.save(path)
    restored = ReplenishmentPPOAgent(env, hidden=16)
    restored.load(path)
    assert np.array_equal(expected, restored.act(obs, greedy=True)[0])
