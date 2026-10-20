"""Regressions for the heuristic's staged weekly price forecast seam."""
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from src.experiments.phase4 import heuristic, make_env


@pytest.mark.parametrize('kind', ['constant', 'ss'])
def test_heuristic_uses_staged_price_at_weekly_boundary(kind):
    env = make_env(days=9)
    env.reset(seed=123)
    env.select_price(4)
    env.replenishment_observation()  # This temporarily pushes, then restores price.
    b = env.observation_components()
    mean = env.sim.mu_for(env.day, env._pending_price)
    position = b['inventory'] + b['pipeline'].sum(axis=1)
    quantity = mean if kind == 'constant' else np.where(
        position < 7 * mean, np.maximum(8 * mean - position, 0), 0)
    expected = np.rint(np.clip(quantity / env.config.replenishment.order_step,
                               0, len(env.grid) - 1)).astype(np.int64)
    assert not np.allclose(mean, b['forecast'])  # Ensure the boundary is exercised.
    np.testing.assert_array_equal(heuristic(env, kind), expected)


@pytest.mark.parametrize('sim', [None, SimpleNamespace(), SimpleNamespace(mu_for=None)])
@pytest.mark.parametrize('pending', [None, np.array([2.])])
def test_generic_forecast_fallback_preserves_grid_and_pipeline(sim, pending):
    env = SimpleNamespace(n=1, sim=sim, _pending_price=pending,
        grid=np.arange(0, 205, 5),
        config=SimpleNamespace(replenishment=SimpleNamespace(order_step=5)),
        observation_components=lambda: dict(forecast=np.array([10.]),
            inventory=np.array([5.]), pipeline=np.array([[10., 0.]])))
    assert heuristic(env, 'constant').tolist() == [2]
    assert heuristic(env, 'ss').tolist() == [13]  # (80 - 15) / 5
    assert heuristic(env, 'zero').tolist() == [0]
    with pytest.raises(ValueError, match='unknown heuristic'):
        heuristic(env, 'invalid')


def test_heuristic_week_two_is_deterministic_without_state_or_rng_mutation():
    from src.env.interface import JointAction
    env = make_env(days=9)
    env.reset(seed=123)
    env.select_price(4)
    for _ in range(7):
        env.step(JointAction(None, np.zeros(env.n, dtype=np.int64)))
    assert env.day == 8 and env.pricing_due
    env.select_price(0)  # Reverse direction at the next weekly boundary.
    env.replenishment_observation()

    def snapshot():
        return pickle.dumps((env.day, env._current_price, env._pending_price,
            env.sim._price, env.forecaster._price, env.sim._env_day,
            env.sim._last_mu, env.sim.base_prices, env.inventory.__dict__,
            env.pipeline.__dict__, env.np_random.bit_generator.state,
            env.sim.rng.bit_generator.state, env.observation_components()))

    before = snapshot()
    mean = env.sim.mu_for(env.day, env._pending_price)
    expected = np.rint(np.clip(mean / env.config.replenishment.order_step,
                               0, len(env.grid) - 1)).astype(np.int64)
    for _ in range(3):
        np.testing.assert_array_equal(heuristic(env, 'constant'), expected)
        heuristic(env, 'ss')
        heuristic(env, 'zero')
    assert snapshot() == before


def test_unstaged_calibrated_forecast_remains_named_forecast():
    env = make_env(days=2, joint=False)
    env.reset(seed=123)
    mean = env.observation_components()['forecast']
    expected = np.rint(np.clip(mean / env.config.replenishment.order_step,
                               0, len(env.grid) - 1)).astype(np.int64)
    np.testing.assert_array_equal(heuristic(env, 'constant'), expected)


def test_noncalibrated_joint_env_keeps_recent_mean_forecast():
    from src.env.joint_env import JointRetailEnv
    env = JointRetailEnv(episode_length_days=2)
    env.reset(seed=123)
    assert not hasattr(env, 'sim')
    env.select_price(4)
    mean = env.observation_components()['forecast']
    expected = np.rint(np.clip(mean / env.config.replenishment.order_step,
                               0, len(env.grid) - 1)).astype(np.int64)
    np.testing.assert_array_equal(heuristic(env, 'constant'), expected)
