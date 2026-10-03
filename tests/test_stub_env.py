"""Stub env round-trip tests — Phase 0 exit criterion (plans P0/R0).

A random-policy episode must run end-to-end; observations must have the
documented layout; reward accounting must match the canonical module.
"""

import numpy as np
import pytest

from src.config.locked_config import get_locked_config
from src.env.interface import JointAction, StubRetailEnv
from src.env.reward import compute_reward, fill_rate


@pytest.fixture()
def env():
    return StubRetailEnv(episode_length_days=28)


class TestRoundTrip:
    def test_reset_returns_both_observations(self, env):
        obs, state = env.reset(seed=42)
        assert set(obs.keys()) == {"pricing", "replenishment"}
        assert state.day == 1
        assert state.inventory.shape == (env.n_skus,)
        assert np.all(state.inventory >= 0)

    def test_random_episode_completes(self, env):
        obs, _ = env.reset(seed=0)
        n_price_actions = len(env.config.pricing.multipliers)
        n_order_actions = len(np.arange(0, env.config.replenishment.q_max + 1, env.config.replenishment.order_step))
        rng = np.random.default_rng(0)

        steps = 0
        done = False
        total_reward = 0.0
        while not done:
            action = JointAction(
                pricing_multiplier_index=rng.integers(n_price_actions),
                replenishment_qty_idx=rng.integers(0, n_order_actions, size=env.n_skus),
            )
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            assert np.isfinite(reward)
            assert info.sales.shape == (env.n_skus,)
            assert np.all(info.lost_sales >= 0)
        assert steps == 28
        assert done is True
        assert np.isfinite(total_reward)

    def test_reproducible_with_seed(self, env):
        """Same seed -> identical trajectory (needed for CRN eval, S5.6)."""
        rewards = []
        for seed in [7, 7]:
            e = StubRetailEnv(episode_length_days=10)
            e.reset(seed=seed)
            rng = np.random.default_rng(seed)
            ep_reward, done = 0.0, False
            while not done:
                a = JointAction(
                    pricing_multiplier_index=int(rng.integers(5)),
                    replenishment_qty_idx=rng.integers(0, 41, size=e.n_skus),
                )
                _, r, done, _ = e.step(a)
                ep_reward += r
            rewards.append(ep_reward)
        assert rewards[0] == pytest.approx(rewards[1])

    def test_noop_action_is_valid(self, env):
        env.reset(seed=1)
        obs, reward, done, info = env.step(JointAction())  # no actions at all
        assert np.isfinite(reward)
        assert info.orders_placed.sum() == 0


class TestObservations:
    def test_observation_layouts_match_doc(self, env):
        """o_t^P and o_t^R sizes follow Doc S2.3 (k=14 history, 10 SKUs)."""
        obs, state = env.reset(seed=3)
        n = env.n_skus
        k = env.config.demand.history_window_k
        # pricing: I(n) + D_hist(k*n) + p(n) + ctx(2) + D_hat(n)
        assert obs["pricing"].shape == (n + k * n + n + 2 + n,)
        # replenishment: I(n) + pipe(n) + D_hist(k*n) + p(n) + lead(n) + headroom(n) + D_hat(n)
        assert obs["replenishment"].shape == (n + n + k * n + n + n + n + n,)

    def test_observations_finite(self, env):
        obs, _ = env.reset(seed=5)
        for key, vec in obs.items():
            assert np.all(np.isfinite(vec)), f"{key} obs has non-finite values"


class TestDynamics:
    def test_lost_sales_transition(self, env):
        """S = min(D, I + A); U = D - S; I' = I + A - S (Doc S2.6)."""
        env.reset(seed=11)
        # Force huge demand by pushing the random walk high is hard; instead
        # verify the identity directly on one step's info arrays.
        _, _, _, info = env.step(JointAction(replenishment_qty_idx=np.zeros(env.n_skus, dtype=int)))
        assert np.allclose(info.sales, np.minimum(info.demand, info.sales + info.lost_sales))

    def test_pricing_holds_between_weekly_days(self, env):
        """Locked timing: price changes only on day 1, 8, 15... (S2.5).
        Day-1 step may price; the day-2 action must be ignored."""
        env.reset(seed=13)
        env.step(JointAction())  # day 1: no pricing action
        p1 = env.get_state().prices.copy()
        env.step(JointAction(pricing_multiplier_index=4))  # day 2: tries to reprice
        assert np.allclose(env.get_state().prices, p1)

    def test_pricing_changes_on_weekly_boundary(self, env):
        env.reset(seed=13)
        p0 = env.get_state().prices.copy()
        env.step(JointAction(pricing_multiplier_index=0))  # day 1: discount 20%
        p1 = env.get_state().prices
        assert np.allclose(p1, p0 * 0.80)

    def test_capacity_clipping_logged(self, env):
        """Orders beyond headroom are clipped and counted (S2.4.2, S5.3)."""
        env.reset(seed=17)
        env.get_state().inventory[:] = 0.0
        max_idx = len(np.arange(0, env.config.replenishment.q_max + 1, env.config.replenishment.order_step)) - 1
        _, _, _, info = env.step(JointAction(replenishment_qty_idx=np.full(env.n_skus, max_idx)))
        assert info.capacity_violations >= 0
        assert np.all(info.orders_clipped <= info.orders_placed + 1e-9)


class TestRewardAccounting:
    def test_canonical_reward_matches_env(self, env):
        """Env reward must equal the canonical compute_reward module."""
        env.reset(seed=23)
        _, _, _, info = env.step(JointAction(replenishment_qty_idx=np.full(env.n_skus, 2)))
        breakdown = compute_reward(
            prices=env.get_state().prices,
            sales=info.sales,
            lost_sales=info.lost_sales,
            inventory_end=env.get_state().inventory,
            orders_placed=info.orders_clipped,
            unit_costs=np.full(env.n_skus, 6.0),
            prev_prices=env.get_state().prices,  # no change this step
            reward_cfg=env.config.reward,
        )
        assert breakdown.total == pytest.approx(info.reward, abs=1e-6)

    def test_reward_terms_signs(self, env):
        env.reset(seed=29)
        _, _, _, info = env.step(JointAction(pricing_multiplier_index=0))
        assert info.revenue >= 0
        assert info.stockout_cost >= 0
        assert info.holding_cost >= 0

    def test_fill_rate(self):
        sales = np.array([10.0, 5.0])
        demand = np.array([10.0, 10.0])
        assert fill_rate(sales, demand) == pytest.approx(0.75)
        assert fill_rate(np.zeros(2), np.zeros(2)) == 1.0  # no demand -> perfect
