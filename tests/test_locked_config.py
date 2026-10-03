"""Config integrity tests — locked decisions must never drift silently."""

import dataclasses

import pytest

from src.config.locked_config import (
    EnvConfig,
    EvaluationConfig,
    LeadTimeConfig,
    LockedConfig,
    PricingActionConfig,
    ReplenishmentActionConfig,
    RewardConfig,
    RLConfig,
    TimingConfig,
    get_locked_config,
)


class TestLockedValues:
    """Values from Doc Table 1 and sections 2.4-2.5."""

    def test_env_scope(self):
        cfg = EnvConfig()
        assert cfg.n_skus == 10
        assert cfg.perishable is False
        assert cfg.lost_sales is True
        assert cfg.shared_capacity is True
        assert cfg.capacity_definition == "on_hand"

    def test_timing_cadence(self):
        cfg = TimingConfig()
        assert cfg.pricing_interval_days == 7
        assert cfg.replenishment_interval_days == 1
        assert cfg.pricing_day_in_cycle == 1  # day-1 price, held days 2-7

    def test_pricing_action_space(self):
        cfg = PricingActionConfig()
        assert cfg.multipliers == (0.80, 0.90, 1.00, 1.10, 1.20)

    def test_replenishment_action_grid(self):
        cfg = ReplenishmentActionConfig()
        assert cfg.order_step == 5
        assert cfg.q_max > 0

    def test_rl_ctde_principles(self):
        cfg = RLConfig()
        assert cfg.gamma == 0.99
        assert cfg.shared_actor_params is False   # heterogeneous actors (S2.10)
        assert cfg.centralized_critic is True     # CTDE (S2.9)
        assert cfg.shared_reward is True          # same scalar to both agents

    def test_evaluation_protocol(self):
        cfg = EvaluationConfig()
        assert cfg.n_training_seeds_dev >= 5
        assert cfg.n_eval_episodes_min >= 50
        assert cfg.common_random_numbers is True


class TestConfigIntegrity:
    def test_all_dataclasses_frozen(self):
        """Locked config must be immutable — accidental edits are bugs."""
        for cls in [EnvConfig, TimingConfig, PricingActionConfig,
                    ReplenishmentActionConfig, LeadTimeConfig,
                    RewardConfig, RLConfig, EvaluationConfig]:
            assert cls.__dataclass_params__.frozen, f"{cls.__name__} not frozen"

    def test_aggregate_returns_singleton_shape(self):
        cfg = get_locked_config()
        assert isinstance(cfg, LockedConfig)
        assert cfg.env.n_skus == 10

    def test_no_unexpected_fields(self):
        """Guard against field additions that skip the design doc update."""
        expected = {
            "EnvConfig", "TimingConfig", "PricingActionConfig",
            "ReplenishmentActionConfig", "LeadTimeConfig", "DemandConfig",
            "RewardConfig", "RLConfig", "EvaluationConfig",
        }
        actual = {f.name for f in dataclasses.fields(LockedConfig)}
        assert actual == {"env", "timing", "pricing", "replenishment",
                          "lead_time", "demand", "reward", "rl", "evaluation"}
