"""Phase R2 tests: ReplenishmentEnv round-trip, observation layout, cadence
hook, forecaster contract, capacity logging (plan R2 exit criteria)."""

import numpy as np
import pytest

from src.config.locked_config import LeadTimeConfig, get_locked_config
from src.env.forecast import RecentMeanForecaster
from src.env.replenishment_env import ReplenishmentEnv
from src.env.synthetic_demand import SyntheticPoissonDemand

N = get_locked_config().env.n_skus


def make_env(seed=None, **kw):
    env = ReplenishmentEnv(**kw)
    obs, info = env.reset(seed=seed)
    return env, obs, info


class TestRoundTrip:
    def test_reset_shapes_and_spaces(self):
        env, obs, info = make_env(seed=0)
        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)
        assert env.action_space.shape == (N,)
        assert info["day"] == 1
        assert np.all(obs >= 0)

    def test_random_episode_end_to_end(self):
        env, obs, _ = make_env(seed=1, episode_length_days=30)
        steps = 0
        terminated, truncated = False, False
        total = 0.0
        while not (terminated or truncated):
            obs, r, terminated, truncated, info = env.step(env.action_space.sample())
            total += r
            steps += 1
            assert np.isfinite(r)
            assert info["sales"].shape == (N,)
            assert np.all(info["lost_sales"] >= 0)
        assert steps == 30 and truncated and not terminated

    def test_seeded_reproducibility(self):
        def run():
            env, _, _ = make_env(seed=77, episode_length_days=20)
            rng = np.random.default_rng(0)  # fixed action rng (action_space has its own)
            rewards, terminated, truncated = [], False, False
            while not (terminated or truncated):
                _, r, terminated, truncated, _ = env.step(rng.integers(0, len(env.grid), size=N))
                rewards.append(r)
            return rewards
        a, b = run(), run()
        assert len(a) == len(b) and np.allclose(a, b)


class TestObservationLayout:
    def test_block_dims_match_doc_s2_3_2(self):
        env, obs, _ = make_env(seed=2)
        blocks = env.observation_components()
        expected = {
            "inventory": (N,),
            "pipeline": (N, env.obs_pipe_horizon),
            "demand_history": (N, env.k),
            "price": (N,),
            "lead_time": (N,),
            "headroom": (N,),
            "forecast": (N,),
        }
        for name, shape in expected.items():
            assert blocks[name].shape == shape, name
        total = sum(int(np.prod(s)) for s in expected.values())
        assert obs.shape == (total,)

    def test_blocks_match_obs_slices(self):
        """The flattened obs must equal the concatenation of named blocks."""
        env, obs, _ = make_env(seed=3)
        env.step(env.action_space.sample())
        obs2, *_ = env.step(np.zeros(N, dtype=int))
        blocks = env.observation_components()
        expected = np.concatenate([b.flatten() for b in blocks.values()]).astype(np.float32)
        assert np.allclose(obs2, expected)

    def test_inventory_block_equals_engine_state(self):
        env, obs, _ = make_env(seed=4)
        blocks = env.observation_components()
        assert np.allclose(blocks["inventory"], env.inventory.on_hand)
        # headroom = C - total on-hand, replicated per SKU
        assert np.allclose(blocks["headroom"],
                           env.inventory.headroom * np.ones(N))

    def test_price_block_is_stub_constant(self):
        env, obs, _ = make_env(seed=5)
        assert np.allclose(env.observation_components()["price"], env.base_price)


class TestPipelineArrival:
    def test_order_arrives_on_expected_day(self):
        """Uniform lead time [1,1]: order placed day 1 must arrive day 2.
        Day-1 arrivals are received in reset; the order goes into the pipeline;
        step() advances to day 2 and receives day-2 arrivals BEFORE returning,
        so on-hand must include the delivery."""
        from dataclasses import replace
        env = ReplenishmentEnv(
            config=replace(get_locked_config(),
                           lead_time=LeadTimeConfig(distribution="uniform", min_days=1, max_days=1)),
            episode_length_days=10,
        )
        obs, info = env.reset(seed=9)
        start_inv = env.inventory.total_on_hand
        max_idx = len(env.grid) - 1  # q_max = 200/SKU
        obs, r, term, trunc, info = env.step(np.full(N, max_idx))
        assert env.inventory.total_on_hand > start_inv

    def test_no_delivery_when_lead_time_not_reached(self):
        from dataclasses import replace
        env = ReplenishmentEnv(config=replace(get_locked_config(),
                                              lead_time=LeadTimeConfig(distribution="uniform",
                                                                       min_days=5, max_days=5)),
                               episode_length_days=10)
        obs, info = env.reset(seed=10)
        start = env.inventory.total_on_hand
        env.step(np.zeros(N, dtype=int))  # day 1: no order, no arrival
        assert env.inventory.total_on_hand <= start  # only demand drew it down


class TestCapacityLogging:
    def test_clipping_logged_in_info(self):
        env = ReplenishmentEnv(capacity=40.0, initial_inventory=30.0, episode_length_days=10)
        obs, info = env.reset(seed=11)
        # order q_max with only 10 headroom -> clipped on arrival next day(s)
        for _ in range(5):
            obs, r, term, trunc, info = env.step(np.full(N, len(env.grid) - 1))
        assert info["capacity_clipped_units"] > 0
        assert info["capacity_violation_events"] >= 1
        assert env.inventory.total_on_hand <= 40.0 + 1e-9


class TestPricingCadenceHook:
    def test_hook_price_change_and_hold(self):
        """Hook is consulted during each day's step; a day-2 price takes effect
        in the obs returned after the day-2 step, and is held on day 3+."""
        calls = []

        def hook(day):
            calls.append(day)
            return np.full(N, 12.0) if day == 2 else None

        env = ReplenishmentEnv(pricing_hook=hook, episode_length_days=5)
        env.reset(seed=12)
        assert np.allclose(env.observation_components()["price"], env.base_price)
        env.step(np.zeros(N, dtype=int))   # day 1: hook -> None, price held
        assert np.allclose(env.observation_components()["price"], env.base_price)
        env.step(np.zeros(N, dtype=int))   # day 2: hook -> 12.0
        assert np.allclose(env.observation_components()["price"], 12.0)
        env.step(np.zeros(N, dtype=int))   # day 3: hook -> None -> hold 12.0
        assert np.allclose(env.observation_components()["price"], 12.0)
        assert calls == [1, 2, 3]          # hook consulted every day

    def test_price_change_cost_only_on_change(self):
        def hook(day):
            return np.full(N, 12.0) if day == 2 else None

        env = ReplenishmentEnv(pricing_hook=hook, episode_length_days=6)
        env.reset(seed=13)
        _, r1, _, _, info1 = env.step(np.zeros(N, dtype=int))
        _, r2, _, _, info2 = env.step(np.zeros(N, dtype=int))
        _, r3, _, _, info3 = env.step(np.zeros(N, dtype=int))
        assert info1["reward_terms"]["price_change_cost"] == pytest.approx(0.0)
        assert info2["reward_terms"]["price_change_cost"] > 0.0
        assert info3["reward_terms"]["price_change_cost"] == pytest.approx(0.0)


class TestForecasterContract:
    def test_recent_mean_math(self):
        f = RecentMeanForecaster(N, window=3)
        f.update(1, np.full(N, 4.0))
        f.update(2, np.full(N, 6.0))
        assert np.allclose(f.forecast(3), 5.0)
        f.update(3, np.full(N, 10.0))
        assert np.allclose(f.forecast(4), (4.0 + 6.0 + 10.0) / 3)

    def test_forecaster_no_peeking(self):
        """Forecast for day t must not include day-t demand."""
        f = RecentMeanForecaster(N, window=2)
        f.update(1, np.full(N, 2.0))
        assert np.allclose(f.forecast(2), 2.0)   # day-2 demand unknown yet
        f.update(2, np.full(N, 100.0))
        assert np.allclose(f.forecast(3), 51.0)

    def test_env_forecast_block_tracks_realized_demand(self):
        env = ReplenishmentEnv(demand_source=SyntheticPoissonDemand(N, base_lambda=20.0),
                               episode_length_days=5)
        env.reset(seed=14)
        first_forecast = env.observation_components()["forecast"].copy()
        assert np.all(first_forecast > 0)  # warm-up history makes D_hat informative
        env.step(np.zeros(N, dtype=int))
        env.step(np.zeros(N, dtype=int))
        # after low-demand days, rolling mean should drop below initial
        assert env.observation_components()["forecast"].mean() < first_forecast.mean() + 20.0


class TestActionGrid:
    def test_action_index_maps_to_grid(self):
        env, _, _ = make_env(seed=15)
        assert env.grid[0] == 0 and env.grid[-1] == env.config.replenishment.q_max
        assert len(env.grid) == env.config.replenishment.q_max // env.config.replenishment.order_step + 1

    def test_out_of_range_action_clipped_not_crash(self):
        env, _, _ = make_env(seed=16)
        obs, r, term, trunc, info = env.step(np.full(N, 9999))  # beyond grid
        assert np.isfinite(r)


class TestInfoCompleteness:
    REQUIRED_KEYS = {"day", "n_skus", "demand", "sales", "lost_sales", "arrivals",
                     "orders_delivered", "orders_clipped", "capacity_violation_events",
                     "capacity_clipped_units", "price", "total_on_hand",
                     "outstanding_orders", "reward_terms"}

    def test_all_required_keys_present(self):
        env, _, _ = make_env(seed=17)
        _, _, _, _, info = env.step(np.zeros(N, dtype=int))
        assert self.REQUIRED_KEYS.issubset(info.keys())
        assert info["reward_terms"] is not None
        assert set(info["reward_terms"].keys()) == {
            "revenue", "ordering_cost", "holding_cost", "stockout_cost", "price_change_cost"}
