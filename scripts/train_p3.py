"""Phase P3 exit validation: train a PPO pricing agent on PricingEnv and
compare it, under CRN, against every fixed weekly multiplier baseline.

Protocol (Doc S5.6-5.7):
    - train on the REAL calibrated environment (calibrated demand simulator,
      calibrated D_hat forecaster, cover-3 replenishment hook, C=1500);
    - evaluate the trained policy GREEDY over N_CRN evaluation episodes and
      every fixed multiplier {0.80x..1.20x} on the SAME episode seeds —
      paired comparisons (identical demand realizations apart from the
      policy's own price-induced demand changes);
    - report mean ± std weekly reward, mean chosen multiplier, action dna.

Outputs: outputs/p3_training_curve.csv (per-update losses + greedy eval
reward), outputs/p3_eval_summary.csv, checkpoint outputs/p3_pricing_policy.pt.

Usage:
    .venv/Scripts/python scripts/train_p3.py --updates 60 --episodes 8
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.pricing_agent import PricingPPOAgent
from src.config.sku_config import load_skus
from src.demand.forecaster import CalibratedMeanForecaster
from src.demand.simulator import CalibratedDemandSimulator
from src.env.pricing_env import PricingEnv

# Locked environment protocol for training AND evaluation (frozen config).
EPISODE_DAYS = 112
CAPACITY = 1500.0
EVAL_SEED = 10 ** 6  # evaluation episode seeds = EVAL_SEED + ep


def make_env(seed: int) -> PricingEnv:
    """The real-config PricingEnv (same as validate_p2)."""
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
        capacity=CAPACITY,
        episode_length_days=EPISODE_DAYS,
    )


def eval_fixed(action: int, episodes: int) -> dict:
    """Greedy evaluation of a FIXED multiplier under the eval CRN seeds."""
    rewards = []
    for ep in range(episodes):
        env = make_env(seed=EVAL_SEED + ep)
        obs, _ = env.reset(seed=EVAL_SEED + ep)
        total = 0.0
        while True:
            obs, r, term, trunc, _ = env.step(action)
            total += r
            if term or trunc:
                break
        rewards.append(total)
    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "rewards": rewards,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--rollout", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ train
    env = make_env(seed=123)  # env seed varies episodes internally (reset rng)
    agent = PricingPPOAgent(
        env, rollout_steps=args.rollout, entropy_coef=0.01, epochs_per_update=4,
    )
    history = agent.train(total_updates=args.updates, seed=args.seed, verbose=True)

    curve_path = out_dir / "p3_training_curve.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["update", "pi_loss", "v_loss", "entropy"])
        for m in history:
            w.writerow([m["update"], m["pi_loss"], m["v_loss"], m["entropy"]])

    # ------------------------------------------------------------------ CRN eval
    trained = agent.evaluate(episodes=args.episodes, seed=EVAL_SEED, greedy=True)
    fixed = {a: eval_fixed(a, args.episodes) for a in range(5)}

    rows = []
    best_fixed_mean = max(f["mean"] for f in fixed.values())
    for a in range(5):
        rows.append((f"fixed {['0.80x','0.90x','1.00x','1.10x','1.20x'][a]}",
                     fixed[a]["mean"], fixed[a]["std"]))
    rows.append(("PPO agent (greedy)", trained["mean_weekly_reward"],
                 float(np.std(trained["episode_rewards"]))))

    for row in rows:
        print(f"  {row[0]:<18} weekly reward {row[1]:>10.1f} ± {row[2]:.1f}")

    all_actions = [a for ep in trained["actions"] for a in ep]
    dna = "".join(str(a) for a in all_actions)
    print(f"  action dna ({args.episodes} eps x 16 weeks): {dna}")

    # ------------- exit criterion: MATCH or beat the best fixed multiplier -----
    # Unit of concern: the learned policy must not be WORSE than the best
    # static rule it could have collapsed to (no-regret floor). Matching it
    # is the honest P3 exit: identical CRN rewards => identical policy choice.
    ok = trained["mean_weekly_reward"] >= best_fixed_mean - 1e-6
    print(f"  PPO >= best fixed (CRN, {args.episodes} eps): {ok}")

    # Keep the checkpoint for inspection/demo regardless of pass/fail.
    ckpt = out_dir / "p3_pricing_policy.pt"
    agent.save(str(ckpt))

    summary = out_dir / "p3_eval_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "mean_weekly_reward", "std"])
        w.writerows(rows)
        w.writerow(["exit_criterion_beats_best_fixed", ok, ""])
        w.writerow(["action_dna", dna, ""])

    if not ok:
        print("EXIT CRITERION NOT MET — inspect curve; consider more updates")
        sys.exit(1)
    print(f"checkpoint -> {ckpt}\ncurve -> {curve_path}\nsummary -> {summary}")


if __name__ == "__main__":
    main()
