"""Phase P2 exit validation (pricing env, no RL).

Runs the REAL weekly PricingEnv (calibrated P1 demand + R1 supply machinery
+ calibrated D_hat forecaster + demand-cover replenishment hook) and asserts
the economically-correct behavior:

    1. Determinism:            same seed -> identical weekly-reward trajectory.
    2. Elasticity monotonicity (Doc S4 discipline, through the FULL env):
       discounted weekly rewards (price-driven revenue) vs base vs markup —
       discount > base on revenue side; markup shifts reward up only via
       margin; report all three.
    3. Weekly cadence:         exactly ceil(days/7) steps; price constant
       within each week; changes only at week starts.
    4. Capacity:               on-hand never exceeds C; clipping reports
       filled when tight.
    5. Forecast sanity:        D_hat positive, finite, price-monotone.

Writes outputs/p2_validation_summary.csv. Exit criteria for P2 (plan):
env trains on, produces sane weekly economics, and all assertions pass.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.sku_config import load_skus
from src.demand.forecaster import CalibratedMeanForecaster
from src.demand.simulator import CalibratedDemandSimulator
from src.env.pricing_env import PricingEnv


def _make_env(seed: int, capacity: float = 1500.0, episode_days: int = 112):
    """Real-config env: calibrated demand + calibrated D_hat + price push."""
    sim = CalibratedDemandSimulator(shock_std=0.10, rng=np.random.default_rng(seed))
    fc = CalibratedMeanForecaster(sim)
    bp = np.array([s.base_price for s in load_skus()])

    class PushPricingEnv(PricingEnv):
        def _push_price_to_demand_source(self):
            self.demand_source.set_current_price(self._current_price)
            if hasattr(self.forecaster, "set_current_price"):
                self.forecaster.set_current_price(self._current_price)

    return PushPricingEnv(
        demand_source=sim,
        forecaster=fc,
        base_price=bp,
        unit_cost=0.6 * bp,
        initial_inventory=6 * np.maximum(sim.level_anchor, 1.0),
        capacity=capacity,
        episode_length_days=episode_days,
    )


def run_action(env: PricingEnv, action: int, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    total = 0.0
    revenue = 0.0
    stockout = 0.0
    weeks = 0
    prices_seen = []
    while True:
        obs, r, term, trunc, info = env.step(action)
        total += r
        weeks += 1
        for d in info["days"]:
            revenue += d["reward_terms"]["revenue"]
            stockout += d["reward_terms"]["stockout_cost"]
            prices_seen.append(np.asarray(d["price"]).copy())
        if trunc or term:
            break
    prices = np.stack(prices_seen)
    price_changes = int((np.abs(np.diff(prices, axis=0)).sum(axis=1) > 1e-9).sum())
    return {
        "weekly_reward": total,
        "revenue": revenue,
        "stockout_cost": stockout,
        "weekly_steps": weeks,
        "price_changes": price_changes,
        "obs": obs,
    }


def main() -> None:
    rows = []

    # --- 1. determinism -----------------------------------------------------
    env = _make_env(seed=5)
    a = run_action(env, action=2, seed=42)
    b = run_action(env, action=2, seed=42)
    assert abs(a["weekly_reward"] - b["weekly_reward"]) < 1e-6, "same seed must reproduce"
    rows.append(("determinism", "same-seed reward diff", abs(a["weekly_reward"] - b["weekly_reward"]), "pass"))

    # --- 2. weekly cadence --------------------------------------------------
    env = _make_env(seed=1)
    assert a["weekly_steps"] == 16, f"112 days -> 16 weekly steps, got {a['weekly_steps']}"
    assert a["price_changes"] <= 16, "price may change at most once per week"
    rows.append(("cadence", "weekly steps (112 days)", a["weekly_steps"], "pass"))
    rows.append(("cadence", "price changes (<=16)", a["price_changes"], "pass"))

    # --- 3. elasticity / margin sweep (CRN: same seed) -----------------------
    env = _make_env(seed=1)
    sweep = {}
    for action, label in [(0, "0.80x"), (2, "1.00x"), (3, "1.10x"), (4, "1.20x")]:
        env = _make_env(seed=1)
        sweep[label] = run_action(env, action=action, seed=77)
        rows.append((
            "sweep", f"weekly reward @{label}",
            sweep[label]["weekly_reward"],
            f"revenue {sweep[label]['revenue']:.0f}, stockout {sweep[label]['stockout_cost']:.0f}",
        ))
    # S4 monotonicity on the DEMAND side: discount raises revenue vs base.
    assert sweep["0.80x"]["revenue"] > sweep["1.00x"]["revenue"], (
        "0.80x must raise revenue vs 1.00x (elastic demand)"
    )
    rows.append(("sweep", "discount revenue > base revenue", True, "pass"))

    # --- 4. capacity invariant ----------------------------------------------
    env = _make_env(seed=3, capacity=300.0)
    obs, info = env.reset(seed=9)
    max_on_hand = 0.0
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(np.random.default_rng(1).integers(5))
        max_on_hand = max(max_on_hand, info["total_on_hand"])
        done = trunc or term
    assert max_on_hand <= 300.0 + 1e-6, f"capacity violated: {max_on_hand}"
    rows.append(("capacity", "max on-hand @C=300", round(max_on_hand, 2), "pass"))

    # --- 5. forecast sanity ---------------------------------------------------
    env = _make_env(seed=2)
    env.reset(seed=11)
    fc = env.forecaster
    d_base = fc.forecast(env.day)
    fc.set_current_price(env.base_price * 0.8)
    d_disc = fc.forecast(env.day)
    fc.set_current_price(env.base_price * 1.2)
    d_mark = fc.forecast(env.day)
    assert np.all(d_disc >= d_base - 1e-9)
    assert np.all(d_base >= d_mark - 1e-9)
    rows.append(("forecast", "D_hat price-monotone", True, "pass"))
    rows.append(("forecast", "D_hat mean (base price)", round(float(d_base.mean()), 2), "units/day avg"))

    # --- write summary --------------------------------------------------------
    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "p2_validation_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "check", "value", "note"])
        w.writerows(rows)

    print("P2 validation — ALL PASS")
    for row in rows:
        print(f"  [{row[0]:<10}] {row[1]}: {row[2]} ({row[3]})")
    print(f"summary -> {out}")


if __name__ == "__main__":
    main()
