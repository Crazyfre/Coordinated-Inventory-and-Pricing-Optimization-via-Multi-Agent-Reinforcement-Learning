"""Phase P1 exit validation (plan P1): does the calibrated simulator
reproduce M5 demand statistics, and is the elasticity slider sane?

Checks:
  1. Backtest: replay HISTORICAL prices day-by-day through the simulator over
     a 365-day train window; compare per-series simulated vs actual mean/std.
  2. Elasticity slider: monotone negative response at the locked elasticity.
  3. Correlated shocks (S5 preview): positive same-day co-deviation.

Usage:  .venv/Scripts/python scripts/validate_p1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.demand.alignment import build_aligned_dataset  # noqa: E402
from src.demand.simulator import CalibratedDemandSimulator  # noqa: E402

N_REPS = 30
WINDOW_DAYS = 365
START_DAY = 1000  # inside the train period


def replay_prices(sim: CalibratedDemandSimulator, aligned, day_lo: int, day_hi: int,
                  price_mult: float = 1.0):
    """Feed the simulator historical prices for each day in the window.

    Base prices are set to the WINDOW MEDIAN per series: this neutralises the
    price-LEVEL shift (locked base prices are last-52-week medians and M5
    prices drift) so the backtest validates seasonality/history/dispersion
    reproduction, with the controlled elasticity responding to WITHIN-window
    price movements only.
    """
    panel = aligned.df.pivot(index="day_index", columns="series_id", values="sell_price")
    prices = panel.loc[day_lo:day_hi, sim.series_ids].to_numpy(float)  # (T, n) — column order = sim order
    median = np.median(prices, axis=0)
    sim._base_price_override = median  # window-relative centring
    draws = []
    for rep in range(N_REPS):
        sim.reset(start_day_index=day_lo, rng=np.random.default_rng(1000 + rep))
        rows = []
        for t, p in enumerate(prices, start=1):
            sim.set_current_price(median * price_mult * (p / median))
            rows.append(sim(t))
        draws.append(np.stack(rows))
    return np.stack(draws)  # (reps, T, n)


def main() -> int:
    aligned = build_aligned_dataset()
    sim = CalibratedDemandSimulator(shock_std=0.0, start_day_index=START_DAY)
    sim.set_current_price(np.full(sim.n, 1.0))

    print("=" * 78)
    print(f"P1 VALIDATION — {N_REPS} replications x {WINDOW_DAYS} days "
          f"(M5 days {START_DAY}..{START_DAY + WINDOW_DAYS - 1})")
    print("=" * 78)

    # --- 1. backtest: simulated vs actual ------------------------------------
    sim_draws = replay_prices(sim, aligned, START_DAY, START_DAY + WINDOW_DAYS - 1)
    actual = aligned.df.pivot(index="day_index", columns="series_id", values="y").loc[
        START_DAY:START_DAY + WINDOW_DAYS - 1, sim.series_ids
    ].to_numpy(float)

    sim_mean = sim_draws.mean(axis=(0, 1))
    sim_std = sim_draws.std(axis=(0, 1))
    act_mean = actual.mean(axis=0)
    act_std = actual.std(axis=0)

    print(f"\n{'series':<24}{'act_mean':>9}{'sim_mean':>9}{'mean_rat':>9}"
          f"{'act_sd':>9}{'sim_sd':>9}{'sd_rat':>9}")
    ratios = []
    for k, sid in enumerate(sim.series_ids):
        mr = sim_mean[k] / max(act_mean[k], 1e-9)
        sr = sim_std[k] / max(act_std[k], 1e-9)
        ratios.append(mr)
        print(f"{sid:<24}{act_mean[k]:>9.2f}{sim_mean[k]:>9.2f}{mr:>9.2f}"
              f"{act_std[k]:>9.2f}{sim_std[k]:>9.2f}{sr:>9.2f}")

    log_ratio = np.abs(np.log(np.array(ratios)))
    within = int((log_ratio < np.log(2.0)).sum())
    print(f"\nseries with mean within factor 2 of actual: {within}/10")
    assert within >= 8, "simulator level calibration too far off"
    assert np.nanmedian(sim_std) > 0

    # --- 2. elasticity monotonicity (locked slider e = -1.5) ------------------
    print("\n--- elasticity slider (S4 preview) ---")
    means = {}
    for mult in (0.8, 1.0, 1.2):
        draws = replay_prices(sim, aligned, START_DAY, START_DAY + WINDOW_DAYS - 1,
                              price_mult=mult)
        means[mult] = draws.mean()
        print(f"  price x{mult:.2f}: mean demand = {means[mult]:.2f}")
    assert means[0.8] > means[1.0] > means[1.2], \
        "demand must decrease monotonically with price at e=-1.5"

    # --- 3. correlated shocks (S5 preview) ------------------------------------
    print("\n--- correlated shocks (S5 preview, shock_std=0.6) ---")
    sim3 = CalibratedDemandSimulator(shock_std=0.6, start_day_index=START_DAY)
    R = sim3.L @ sim3.L.T
    np.fill_diagonal(R, 0)
    i, j = np.unravel_index(np.argmax(R), R.shape)
    print(f"  most-correlated pair: {sim3.series_ids[i]} & {sim3.series_ids[j]} "
          f"(target corr {R[i, j]:+.2f})")
    sim3.set_current_price(np.full(sim3.n, 1.0))
    devs = []
    for rep in range(20):
        sim3.reset(start_day_index=START_DAY, rng=np.random.default_rng(900 + rep))
        draws = np.stack([sim3(d) for d in range(1, 200)])
        mu0 = draws.mean(axis=0)
        devs.append((draws[:, i] - mu0[i]) * (draws[:, j] - mu0[j]))
    co_dev = np.mean(devs)
    print(f"  mean same-day co-deviation: {co_dev:+.2f} (positive => correlated)")
    assert co_dev > 0

    print("\nALL P1 VALIDATION ASSERTIONS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
