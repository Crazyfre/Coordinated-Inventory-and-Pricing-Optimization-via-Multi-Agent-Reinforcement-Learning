import numpy as np


def test_calibrated_joint_factory_runs_and_isolates_demand_stream():
    from src.experiments.phase4 import make_env, evaluate
    env = make_env(days=9)
    a = evaluate(env, lambda e,o: np.zeros(e.n,dtype=int), episodes=2, seed=10)
    b = evaluate(env, lambda e,o: np.ones(e.n,dtype=int)*40, episodes=2, seed=10)
    assert a['demands'] == b['demands']
    assert a['fill_rate'] >= 0
    assert np.isfinite(b['mean_daily_reward'])

def test_joint_context_matches_calibrated_calendar():
    from src.experiments.phase4 import make_env
    env = make_env(days=9)
    env.reset(seed=1)
    expected = env.sim.calendar_feats[env.sim.start_day_index-1]
    assert np.array_equal(env.context(),expected)


def test_joint_info_reports_current_day_deliveries():
    from dataclasses import replace
    from src.config.locked_config import get_locked_config
    from src.env.joint_env import JointRetailEnv
    cfg = get_locked_config()
    cfg = replace(cfg,lead_time=replace(cfg.lead_time,min_days=2,max_days=2))
    env = JointRetailEnv(config=cfg,episode_length_days=4)
    env.reset(seed=2)
    env.step(JointAction(None,np.ones(10,dtype=int)))
    _,_,_,info2 = env.step(JointAction(None,np.zeros(10,dtype=int)))
    assert info2['orders_delivered'].sum() == 0
    _,_,_,info3 = env.step(JointAction(None,np.zeros(10,dtype=int)))
    assert info3['orders_delivered'].sum() == 50
    assert info3['day'] == 3


from src.env.interface import JointAction


def test_joint_daily_cadence_and_new_price_observation():
    from src.env.joint_env import JointRetailEnv
    env = JointRetailEnv(episode_length_days=9)
    obs, _ = env.reset(seed=3)
    env.select_price(4)
    assert np.allclose(env.replenishment_observation()[220:230], 12)
    prices = []
    for day in range(1,10):
        obs, reward, done, info = env.step(JointAction(0, np.zeros(10,dtype=int)))
        prices.append(info['price'][0])
        assert info['total_on_hand'] <= env.capacity
    assert prices == [8]*7 + [8]*2
    assert done
