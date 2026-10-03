"""Locked Version 1 configuration (Doc section 1.1, Tables 1-3).

Every dataclass here encodes a decision locked in the research design.
Do not modify values without updating the design document first:
these constants must be identical across all methods in the final
evaluation (Doc sections 2.4.2, 2.7, 4.6).

References like (T1) point to Table 1; (S2.x) to design-document sections.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnvConfig:
    """Locked environment scope (T1)."""

    n_skus: int = 10
    perishable: bool = False          # non-perishable SKUs (T1)
    lost_sales: bool = True           # lost-sales model, no backlogging (T1, S2.6)
    shared_capacity: bool = True      # shared warehouse capacity (T1, S4.6)
    # Capacity convention: constraint applies to ON-HAND inventory (S4.6).
    # sum_i I_(i,t) <= C. Same convention for every method.
    capacity_definition: str = "on_hand"


@dataclass(frozen=True)
class TimingConfig:
    """Multi-timescale decision cadence (S2.5)."""

    pricing_interval_days: int = 7    # pricing refreshed weekly
    replenishment_interval_days: int = 1  # replenishment daily
    # Locked timing convention: day 1 = price selected + order allowed;
    # days 2-7 = price held, replenishment daily; revise at next boundary.
    pricing_day_in_cycle: int = 1


@dataclass(frozen=True)
class PricingActionConfig:
    """Discrete weekly pricing action space (S2.4.1, Table 3)."""

    multipliers: tuple[float, ...] = (0.80, 0.90, 1.00, 1.10, 1.20)
    # p_(i,t) = p_base_i * m_(i,t)


@dataclass(frozen=True)
class ReplenishmentActionConfig:
    """Discrete daily replenishment action space (S2.4.2)."""

    order_step: int = 5               # A_R = {0, 5, 10, 15, 20, ..., Q_max}
    q_max: int = 200                  # per-SKU upper bound (calibrate before final eval)
    # Feasibility (capacity/supplier constraints) is enforced by the simulator
    # via documented clipping, fixed across all methods (S2.4.2 note).


@dataclass(frozen=True)
class LeadTimeConfig:
    """Stochastic bounded supplier lead time (T1)."""

    distribution: str = "truncnormal"  # "truncnormal" | "uniform"
    min_days: int = 2                  # bounded support [min, max]
    max_days: int = 7
    # Optional distribution shape overrides; None -> derived from bounds:
    #   mean = (min + max) / 2, std = (max - min) / 3
    mean_days: float | None = None
    std_days: float | None = None


@dataclass(frozen=True)
class DemandConfig:
    """Demand simulator knobs (S4.4-4.5).

    NOTE: price elasticity is a controlled negative response with a slider,
    NOT the fitted M5 coefficient (causal identification caveat, S4.5).
    """

    count_distribution: str = "negative_binomial"
    history_window_k: int = 14         # demand history D_(t-k:t-1) in observations
    base_elasticity: float = -1.5      # controlled value; sweep for scenario S4
    elasticity_range: tuple[float, float] = (-3.0, -0.5)


@dataclass(frozen=True)
class RewardConfig:
    """Shared economic reward coefficients (S2.7).

    r_t = Revenue - OrderingCost - HoldingCost - StockoutCost - PriceChangeCost
    Weight calibration happens before the final policy comparison and is
    then frozen across algorithms (anti-bias rule, S2.7).
    """

    holding_cost_per_unit: float = 0.02     # h_i, per unit per day (placeholder)
    stockout_cost_per_unit: float = 1.00    # b_i, per unit of unmet demand
    price_change_cost: float = 0.05         # c_p,i per unit absolute price delta
    # Ordering cost c_o,i is per-SKU unit procurement cost; lives in SKU data.
    # Service-level: penalty formulation first; Lagrangian fallback if unstable (S2.8).
    service_target_fill_rate: float = 0.95


@dataclass(frozen=True)
class RLConfig:
    """Algorithm and MDP hyperparameters (S2.9-2.10, T1)."""

    gamma: float = 0.99
    # Heterogeneous MAPPO, CTDE: separate actors (NO parameter sharing across
    # pricing/replenishment), shared centralized critic on global state (S2.10).
    shared_actor_params: bool = False
    centralized_critic: bool = True
    shared_reward: bool = True         # both agents receive the same scalar r_t


@dataclass(frozen=True)
class EvaluationConfig:
    """Experimental protocol (S5.5-5.7, T6)."""

    n_training_seeds_dev: int = 5      # development runs
    n_training_seeds_final: int = 10   # final reported experiment (if feasible)
    n_eval_episodes_min: int = 50
    n_eval_episodes_preferred: int = 100
    common_random_numbers: bool = True  # paired CRN evaluation (S5.6)


@dataclass(frozen=True)
class LockedConfig:
    """Aggregate handle for all locked decisions."""

    env: EnvConfig = field(default_factory=EnvConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    pricing: PricingActionConfig = field(default_factory=PricingActionConfig)
    replenishment: ReplenishmentActionConfig = field(default_factory=ReplenishmentActionConfig)
    lead_time: LeadTimeConfig = field(default_factory=LeadTimeConfig)
    demand: DemandConfig = field(default_factory=DemandConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def get_locked_config() -> LockedConfig:
    """Single entry point. Both tracks import this; never re-instantiate."""
    return LockedConfig()
