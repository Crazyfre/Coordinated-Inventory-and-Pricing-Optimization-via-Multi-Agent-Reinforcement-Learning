"""Abstract RetailEnv interface — the Phase 0 contract.

Both tracks (pricing and replenishment) code against this interface until
integration (Phase P4/R4). Signatures agreed in Phase 0; changing them
requires sign-off from both tracks.

Design notes
------------
- The environment advances in DAILY steps (Doc S2.5).
- Pricing actions arrive weekly, replenishment actions daily. In the stub
  both are accepted every step; the real env (integration) will mask/hold
  pricing on non-weekly days per the locked timing convention.
- `step` accepts a dict of named agent actions so tracks can pass only
  their own action; missing agents fall back to "no-op" defaults.
- Observations are returned as a dict keyed by agent name, so each track
  consumes exactly its own observation (o_t^P or o_t^R, Doc S2.3).

References: (Sx.y) = design doc section, (Tn) = table n.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.config.locked_config import LockedConfig, get_locked_config


# ---------------------------------------------------------------------------
# Action / observation container types
# ---------------------------------------------------------------------------


def pricing_indices(action, scope, n_skus, n_multipliers):
    """Validate locked-grid indices; scalars explicitly mean uniform pricing.

    Vector actions are available only for per_sku scope. No clipping/coercion
    of float/bool indices; caller handles ignored nondecision days first.
    """
    if scope not in ('shared', 'per_sku'):
        raise ValueError("pricing_scope must be 'shared' or 'per_sku'")
    array = np.asarray(action)
    if array.dtype.kind not in 'iu':
        raise ValueError('price indices must be integers')
    if array.ndim == 0:
        indices = np.full(n_skus, int(array), dtype=np.int64)
    elif scope == 'per_sku' and array.shape == (n_skus,):
        indices = array.copy()
    else:
        raise ValueError('price indices require scalar or per-SKU vector shape (n,)')
    if np.any(indices < 0) or np.any(indices >= n_multipliers):
        raise ValueError('price index outside locked grid')
    return indices


@dataclass
class JointAction:
    """Actions for one environment day (S2.4, S2.5).

    pricing_multiplier_index: scalar shared index or per-SKU integer vector
        into PricingActionConfig.multipliers,
        or None on days when the pricing agent does not act.
    replenishment_qty_idx: per-SKU index into the order grid, or None when
        the replenishment agent does not act. The grid itself is derived
        from ReplenishmentActionConfig (0..q_max step order_step).
    """

    pricing_multiplier_index: int | np.ndarray | None = None
    replenishment_qty_idx: np.ndarray | None = None  # shape (n_skus,), int


@dataclass
class EnvState:
    """Global state s_t components (S2.2, Table 2)."""

    day: int                                   # environment day counter, 1-based
    inventory: np.ndarray                      # I_t, shape (n_skus,)
    pipeline: dict[int, dict[int, float]]      # {arrival_day: {sku: qty}}
    demand_history: np.ndarray                 # shape (k, n_skus), most recent last
    prices: np.ndarray                         # p_t, shape (n_skus,)
    lead_time_info: dict[str, Any]             # L_t summary
    capacity: float                            # C
    context: dict[str, Any]                    # x_t calendar/promo features


@dataclass
class StepInfo:
    """Rich per-step info: everything metrics modules need (S5.2-5.4)."""

    demand: np.ndarray                # D_(i,t) realized
    sales: np.ndarray                 # S_(i,t) = min(D, I + A)
    lost_sales: np.ndarray            # U_(i,t)
    arrivals: np.ndarray              # A_(i,t) received this day
    orders_placed: np.ndarray         # q_(i,t) ordered this day (pre-clip)
    orders_clipped: np.ndarray        # q after feasibility enforcement
    revenue: float
    ordering_cost: float
    holding_cost: float
    stockout_cost: float
    price_change_cost: float
    reward: float                     # shared scalar r_t (S2.7)
    capacity_violations: int          # clipped-action events (S5.3)
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class RetailEnv(ABC):
    """Abstract contract for the retail environment (cooperative Markov game,
    Doc S2.1). Tracks implement/extend this; the Phase 0 stub provides a
    minimal random-walk demand version with no pipeline dynamics."""

    def __init__(self, config: LockedConfig | None = None):
        self.config = config or get_locked_config()

    @abstractmethod
    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], EnvState]:
        """Reset to day 1. Returns (observations by agent, global state)."""

    @abstractmethod
    def step(self, action: JointAction) -> tuple[dict[str, np.ndarray], float, bool, StepInfo]:
        """Advance one day.

        Returns:
            obs:    dict with keys "pricing" and/or "replenishment"
            reward: shared scalar r_t (S2.7)
            done:   episode end
            info:   StepInfo for metrics/eval
        """

    @abstractmethod
    def get_state(self) -> EnvState:
        """Full global state (centralized-critic input, S2.9)."""

    @property
    @abstractmethod
    def n_skus(self) -> int:
        """Number of SKUs."""

    # -- observation builders (each track consumes only its own) ------------

    @abstractmethod
    def pricing_observation(self, state: EnvState) -> np.ndarray:
        """o_t^P = [I_t, D_(t-k:t-1), p_(t-1), x_t, D_hat_t] (S2.3.1)."""

    @abstractmethod
    def replenishment_observation(self, state: EnvState) -> np.ndarray:
        """o_t^R = [I_t, Q_t^pipe, D_(t-k:t-1), p_t, L_t, C, D_hat_t] (S2.3.2)."""


# ---------------------------------------------------------------------------
# Phase 0 stub environment
# ---------------------------------------------------------------------------

class StubRetailEnv(RetailEnv):
    """Minimal daily env: random-walk demand, no order pipeline, no lead times.

    Purpose (plans P0/R0): let both tracks build and smoke-test agents,
    observations, reward accounting and PPO loops BEFORE the real inventory
    dynamics (R1) and calibrated demand (P1) exist. Deliberately simple;
    NOT used in any final evaluation.

    Dynamics:
        D_(i,t) ~ Poisson(lambda_i) with lambda a slow random walk
        Orders arrive instantly (lead time = 0) so RL loops are exercisable.
        Capacity: orders clipped so on-hand stays under C (logged, not enforced
        on arrivals — no pipeline exists yet).
    """

    def __init__(self, config: LockedConfig | None = None, episode_length_days: int = 84):
        super().__init__(config)
        self.episode_length_days = episode_length_days
        self.rng = np.random.default_rng()
        self._state: EnvState | None = None
        self._base_prices = np.full(self.config.env.n_skus, 10.0)
        self._base_demand = np.full(self.config.env.n_skus, 10.0)
        self._unit_costs = np.full(self.config.env.n_skus, 6.0)

    # -- RetailEnv protocol --------------------------------------------------

    @property
    def n_skus(self) -> int:
        return self.config.env.n_skus

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], EnvState]:
        self.rng = np.random.default_rng(seed)
        self._state = EnvState(
            day=1,
            inventory=self._base_demand.copy() * 3,  # start with some stock
            pipeline={},
            demand_history=np.tile(self._base_demand, (self.config.demand.history_window_k, 1)),
            prices=self._base_prices.copy(),
            lead_time_info={"mean": 0, "pending_orders": 0},
            capacity=float(self.config.env.n_skus * 50),
            context={"weekday": 0, "week_of_year": 1, "promo": np.zeros(self.n_skus)},
        )
        return self._observations(self._state), self._state

    def step(self, action: JointAction) -> tuple[dict[str, np.ndarray], float, bool, StepInfo]:
        assert self._state is not None, "call reset() before step()"
        cfg = self.config

        # 1. Price for today: new action on pricing days, else hold (S2.5).
        is_pricing_day = (self._state.day - cfg.timing.pricing_day_in_cycle) % cfg.timing.pricing_interval_days == 0
        prev_prices = self._state.prices.copy()
        if action.pricing_multiplier_index is not None and is_pricing_day:
            mult = cfg.pricing.multipliers[action.pricing_multiplier_index]
            self._state.prices = self._base_prices * mult
        price_change_cost = float(cfg.reward.price_change_cost * np.abs(self._state.prices - prev_prices).sum())

        # 2. Replenishment: instant arrival, clipped to capacity headroom.
        orders = np.zeros(self.n_skus)
        clipped_orders = np.zeros(self.n_skus)
        capacity_violations = 0
        if action.replenishment_qty_idx is not None:
            grid = np.arange(0, cfg.replenishment.q_max + 1, cfg.replenishment.order_step)
            orders = grid[np.clip(action.replenishment_qty_idx, 0, len(grid) - 1)].astype(float)
            headroom = max(self._state.capacity - float(self._state.inventory.sum()), 0.0)
            clipped_orders = np.minimum(orders, headroom / self.n_skus)  # naive share; real policy in R1
            capacity_violations = int((clipped_orders < orders).sum())
            self._state.inventory = self._state.inventory + clipped_orders

        # 3. Demand: Poisson random walk (stub stand-in for the calibrated NB
        #    simulator, which plugs in at integration from the pricing track).
        self._base_demand = np.clip(
            self._base_demand * self.rng.normal(1.0, 0.03, size=self.n_skus), 1.0, 50.0
        )
        demand = self.rng.poisson(self._base_demand)

        # 4. Lost-sales transition (S2.6).
        sales = np.minimum(demand, self._state.inventory)
        lost = demand - sales
        self._state.inventory = self._state.inventory - sales

        # 5. Shared reward (S2.7).
        revenue = float((self._state.prices * sales).sum())
        ordering_cost = float((self._unit_costs * clipped_orders).sum())
        holding_cost = float(cfg.reward.holding_cost_per_unit * self._state.inventory.sum())
        stockout_cost = float(cfg.reward.stockout_cost_per_unit * lost.sum())
        reward = revenue - ordering_cost - holding_cost - stockout_cost - price_change_cost

        # 6. Bookkeeping.
        self._state.demand_history = np.vstack([self._state.demand_history[1:], demand])
        self._state.day += 1
        done = self._state.day > self.episode_length_days

        info = StepInfo(
            demand=demand, sales=sales, lost_sales=lost,
            arrivals=clipped_orders, orders_placed=orders, orders_clipped=clipped_orders,
            revenue=revenue, ordering_cost=ordering_cost, holding_cost=holding_cost,
            stockout_cost=stockout_cost, price_change_cost=price_change_cost,
            reward=reward, capacity_violations=capacity_violations,
        )
        return self._observations(self._state), reward, done, info

    def get_state(self) -> EnvState:
        assert self._state is not None, "call reset() first"
        return self._state

    # -- observations ---------------------------------------------------------

    def pricing_observation(self, state: EnvState) -> np.ndarray:
        """o_t^P — inventory, demand history (flattened), prev price, context stub,
        demand forecast stub (recent mean). Real D_hat_t plugs in at integration."""
        d_hat = state.demand_history.mean(axis=0)
        ctx = np.array([state.context["weekday"], state.context["week_of_year"]], dtype=float)
        return np.concatenate([state.inventory, state.demand_history.flatten(), state.prices, ctx, d_hat])

    def replenishment_observation(self, state: EnvState) -> np.ndarray:
        """o_t^R — inventory, pipeline (stub zeros), demand history, price,
        lead-time stub, capacity headroom, forecast stub."""
        d_hat = state.demand_history.mean(axis=0)
        pipeline_stub = np.zeros(self.n_skus)
        lead_stub = np.zeros(self.n_skus)
        headroom = np.full(self.n_skus, state.capacity - state.inventory.sum())
        return np.concatenate([state.inventory, pipeline_stub, state.demand_history.flatten(), state.prices, lead_stub, headroom, d_hat])

    # -- helpers ---------------------------------------------------------------

    def _observations(self, state: EnvState) -> dict[str, np.ndarray]:
        return {
            "pricing": self.pricing_observation(state),
            "replenishment": self.replenishment_observation(state),
        }
