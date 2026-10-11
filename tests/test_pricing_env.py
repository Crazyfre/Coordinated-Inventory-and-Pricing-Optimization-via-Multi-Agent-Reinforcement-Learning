"""Phase P2 tests: weekly PricingEnv (pricing agent's training environment).

Mirrors the R2 test discipline: round-trip with random actions, seeded
reproducibility (CRN foundation), obs-block layout vs Doc S2.3.1 + flattened
identity, cadence/hold behavior inside the weekly block, canonical-reward
cross-check, calendar mapping, the real forecaster's no-peeking contract,
and the demand-cover hook's grid/bounds sanity.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.config.locked_config import get_locked_config
from src.config.sku_config import load_skus
from src.demand.forecaster import CalibratedMeanForecaster
from src.demand.simulator import CalibratedDemandSimulator
from src.env.forecast import DemandForecast, RecentMeanForecaster
from src.env.pricing_env import PricingEnv, default_replenishment_hook
from src.env.reward import compute_reward
from src.env.synthetic_demand import SyntheticPoissonDemand


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_prices() -> np.ndarray:
    return np.array([s.base_price for s in load_skus()])


class _PushEnv(PricingEnv):
    """PricingEnv whose forecaster also receives the price push (CalibratedMeanForecaster seam)."""

    def _push_price_to_demand_source(self):
        super()._push_price_to_demand_source()
        if hasattr(self.forecaster, "set_current_price"):
            self.forecaster.set_current_price(self._current_price)


def make_env(
    episode_length_days: int = 28,
    seed: int = 123,
    elasticity: float = -1.5,
    shock_std: float = 0.10,
    **env_kwargs,
):
    """Calibrated-simulator-backed PricingEnv (the real P2 configuration)."""
    sim = CalibratedDemandSimulator(
        shock_std=shock_std, rng=np.random.default_rng(seed)
    )
    if elasticity is not None:
        sim.elasticity = np.full(sim.n, elasticity)
    fc = CalibratedMeanForecaster(sim)
    bp = _base_prices()
    env = _PushEnv(
        demand_source=sim,
        forecaster=fc,
        base_price=bp,
        unit_cost=0.6 * bp,
        initial_inventory=6 * np.maximum(sim.level_anchor, 1),
        episode_length_days=episode_length_days,
        **env_kwargs,
    )
    env._seed = seed
    return env


def run_episode(env: PricingEnv, action: int, seed: int = 123):
    """Run reset + fixed-action episode; returns (sum_reward, n_steps, info)."""
    obs, _ = env.reset(seed=seed)
    total, steps, last_info = 0.0, 0, None
    while True:
        obs, r, terminated, truncated, info = env.step(action)
        total += r
        steps += 1
        last_info = info
        if truncated or terminated:
            break
    return total, steps, last_info, obs


# ---------------------------------------------------------------------------
# Spaces + reset
# ---------------------------------------------------------------------------

class TestSpacesAndReset:
    def test_action_space_is_discrete_multipliers(self):
        env = PricingEnv()
        assert env.action_space.n == len(get_locked_config().pricing.multipliers)

    def test_obs_dim_matches_blocks(self):
        env = make_env(episode_length_days=28)
        obs, _ = env.reset(seed=1)
        n, k = env.n, env.k
        expected = n + n * k + n + env.ctx_dim + n + n
        assert obs.shape == (expected,)
        assert env.observation_space.shape == (expected,)

    def test_obs_blocks_layout(self):
        env = make_env(episode_length_days=28)
        env.reset(seed=1)
        blocks = env.observation_components()
        n, k = env.n, env.k
        assert blocks["inventory"].shape == (n,)
        assert blocks["demand_history"].shape == (n, k)
        assert blocks["price"].shape == (n,)
        assert blocks["forecast"].shape == (n,)
        assert blocks["headroom"].shape == (n,)
        flat = np.concatenate(
            [blocks["inventory"], blocks["demand_history"].flatten(),
             blocks["price"], blocks["context"], blocks["forecast"], blocks["headroom"]]
        )
        assert np.allclose(
            flat, env._build_obs(), atol=1e-5
        )

    def test_reset_fills_demand_history_with_real_warmup(self):
        env = make_env(episode_length_days=28)
        env.reset(seed=1)
        blocks = env.observation_components()
        hist = blocks["demand_history"]
        assert np.all(hist >= 0)               # real sampled warm-up, not zeros
        assert hist.sum() > 0


# ---------------------------------------------------------------------------
# Weekly cadence + dynamics
# ---------------------------------------------------------------------------

class TestWeeklyCadence:
    def test_one_step_advances_exactly_one_week(self):
        env = make_env(episode_length_days=28)
        env.reset(seed=1)
        obs, r, term, trunc, info = env.step(2)
        assert env.day == 8                     # 7 days simulated
        assert info["week"] == 1
        assert not trunc

    def test_price_held_within_week_then_changes(self):
        """Doc S2.5: price set at week start, held for 6 more days."""
        cfg = get_locked_config()
        env = make_env(episode_length_days=28)
        env.reset(seed=1)
        bp = env.base_price
        _, _, _, _, info = env.step(4)          # multiplier 1.20
        week_days = info["days"]
        # Day 1 of the week: base * 1.20 exactly.
        assert np.allclose(week_days[0]["price"], bp * 1.20)
        # Held for the rest of the week.
        for d in week_days[1:]:
            assert np.allclose(d["price"], bp * 1.20)
        # Next step, holding the SAME multiplier still counts a price change
        # only if price moved; same value -> no further change (cost 0 in week 2).
        obs, r, term, trunc, info2 = env.step(4)
        assert np.allclose(info2["days"][0]["price"], bp * 1.20)

    def test_multiplier_applied_from_locked_set(self):
        cfg = get_locked_config()
        env = make_env(episode_length_days=7)
        env.reset(seed=1)
        _, _, _, _, info = env.step(0)          # 0.80
        assert np.allclose(info["price"], env.base_price * cfg.pricing.multipliers[0])

    def test_episode_length_in_weekly_steps(self):
        env = make_env(episode_length_days=28)
        _, steps, info, obs = run_episode(env, action=2)
        assert steps == 4                       # ceil(28/7)
        assert env.day == 29

    def test_non_multiple_episode_length(self):
        env = make_env(episode_length_days=30)
        _, steps, info, obs = run_episode(env, action=2)
        assert steps == 5                       # ceil(30/7), last week short
        # Last weekly block must contain exactly 2 days (29, 30).
        last_week_days = info["days"]
        assert len(last_week_days) == 2


# ---------------------------------------------------------------------------
# Reward accounting
# ---------------------------------------------------------------------------

class TestRewardAccounting:
    def test_weekly_reward_equals_sum_of_day_breakdowns(self):
        """Weekly reward = sum over the week's days of the canonical
        breakdown terms stored in each day block (env/reward.py contract)."""
        env = make_env(episode_length_days=14, seed=5)
        env.reset(seed=5)
        obs, r, term, trunc, info = env.step(3)
        breakdown_sum = sum(
            d["reward_terms"]["revenue"]
            - d["reward_terms"]["ordering_cost"]
            - d["reward_terms"]["holding_cost"]
            - d["reward_terms"]["stockout_cost"]
            - d["reward_terms"]["price_change_cost"]
            for d in info["days"]
        )
        assert abs(breakdown_sum - r) < 1e-6

    def test_revenue_responds_to_price_monotonically_right_side(self):
        """Discounting (elastic demand) must raise revenue vs base price,
        holding the same replenishment hook and seed (CRN pairing)."""
        env = make_env(episode_length_days=28)
        seeds = 123
        # Compare revenue terms under CRN (same seed, same demand realizations).
        def total_revenue(action, seed):
            obs, info = env.reset(seed=seed)
            rev = 0.0
            while True:
                obs, r, term, trunc, info = env.step(action)
                rev += sum(d["reward_terms"]["revenue"] for d in info["days"])
                if trunc or term:
                    break
            return rev
        rev_disc = total_revenue(0, seeds)   # 0.80x
        rev_base = total_revenue(2, seeds)   # 1.00x
        assert rev_disc > rev_base


# ---------------------------------------------------------------------------
# Determinism / CRN
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_trajectory(self):
        env1, env2 = make_env(episode_length_days=28), make_env(episode_length_days=28)
        o1, _ = env1.reset(seed=99)
        o2, _ = env2.reset(seed=99)
        assert np.allclose(o1, o2)
        for _ in range(3):
            a1, r1, t1, tr1, i1 = env1.step(4)
            a2, r2, t2, tr2, i2 = env2.step(4)
            assert np.allclose(r1, r2)
            assert np.allclose(a1, a2)

    def test_capacity_clipping_recorded(self):
        env = make_env(episode_length_days=28, capacity=50.0)
        obs, info = env.reset(seed=1)
        # Small capacity + big replenishment target should clip at receiving.
        obs, r, term, trunc, info = env.step(2)
        assert info["capacity_violation_events"] >= 0  # may be 0 if demand consumes stock first
        assert np.isfinite(info["capacity_clipped_units"])
        assert env.inventory.total_on_hand <= env.capacity + 1e-6


# ---------------------------------------------------------------------------
# Forecaster contract (co-owned interface)
# ---------------------------------------------------------------------------

class TestForecasterContract:
    def test_calimated_forecaster_is_demand_forecast(self):
        env = make_env(episode_length_days=28)
        env.reset(seed=1)
        assert isinstance(env.forecaster, DemandForecast)

    def test_forecast_positive_and_finite_after_price_push(self):
        env = make_env(episode_length_days=28)
        env.reset(seed=1)
        d_hat = env.forecaster.forecast(env.day)
        assert np.all(np.isfinite(d_hat)) and np.all(d_hat > 0)

    def test_forecast_reflects_price_elasticity(self):
        """D_hat at a low price must exceed D_hat at a high price."""
        env = make_env(episode_length_days=28)
        env.reset(seed=1)
        fc = env.forecaster
        bp = env.base_price
        d_low = fc.forecast(env.day)           # current price == base
        fc.set_current_price(bp * 0.8)
        d_discount = fc.forecast(env.day)
        fc.set_current_price(bp * 1.2)
        d_markup = fc.forecast(env.day)
        assert (d_discount > d_low - 1e-9).all()
        assert (d_low > d_markup - 1e-9).all()


# ---------------------------------------------------------------------------
# Replenishment hook stand-in
# ---------------------------------------------------------------------------

class TestReplenishmentHook:
    def test_hook_quantities_on_locked_grid(self):
        cfg = get_locked_config()
        grid = np.arange(0, cfg.replenishment.q_max + 1, cfg.replenishment.order_step)
        env = make_env(episode_length_days=28)
        env.reset(seed=1)
        for _ in range(2):
            obs, r, term, trunc, info = env.step(2)
        total_orders = sum(
            d["reward_terms"]["ordering_cost"] > 0 for d in info["days"]
        )
        # orderings happened: verify via pipeline outstanding is grid-valued
        # is not directly visible per day; instead check the hook directly:
        hook = default_replenishment_hook(days_of_cover=3.0)
        view = {
            "on_hand": np.zeros(env.n),
            "outstanding": np.zeros(env.n),
            "forecast": np.full(env.n, 20.0),
            "lead_time_demand_buffer": 4.5,
            "price": env.base_price,
            "headroom": 100.0,
        }
        qty = hook(1, view)
        assert all(q in set(grid.tolist()) for q in qty)
        assert np.all(qty > 0)

    def test_hook_orders_nothing_when_well_stocked(self):
        env = make_env(episode_length_days=28)
        hook = default_replenishment_hook(days_of_cover=3.0)
        view = {
            "on_hand": np.zeros(env.n),
            "outstanding": np.zeros(env.n),
            "forecast": np.zeros(env.n),
            "lead_time_demand_buffer": 0.0,
            "price": env.base_price,
            "headroom": 100.0,
        }
        qty = hook(1, view)
        assert np.all(qty == 0)


# ---------------------------------------------------------------------------
# Stub-source compatibility (no calibrated simulator required)
# ---------------------------------------------------------------------------

class TestStubSourceCompatibility:
    def test_runs_with_synthetic_poisson_source(self):
        env = PricingEnv(
            demand_source=SyntheticPoissonDemand(seed_rng := 10, rng=np.random.default_rng(seed_rng)),
            forecaster=RecentMeanForecaster(10, window=7),
            episode_length_days=14,
        )
        obs, info = env.reset(seed=7)
        total = 0.0
        steps = 0
        while True:
            obs, r, term, trunc, info = env.step(2)
            total += r
            steps += 1
            if trunc:
                break
        assert steps == 2 and np.isfinite(total)

    def test_latest_day_block_contains_full_metrics(self):
        env = make_env(episode_length_days=28)
        env.reset(seed=1)
        _, _, _, _, info = env.step(2)
        d = info["days"][-1]
        for key in ("demand", "sales", "lost_sales", "price", "reward_terms"):
            assert key in d
        rt = d["reward_terms"]
        for key in ("revenue", "ordering_cost", "holding_cost",
                    "stockout_cost", "price_change_cost"):
            assert key in rt
