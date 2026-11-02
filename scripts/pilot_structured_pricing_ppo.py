"""Bounded pricing PPO pilot; frozen ordering, training-calendar scenarios only."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.algos.parameter_policy_gradient import order_action
from src.algos.structured_pricing_ppo import StructuredPricingPPO, pricing_features, critic_features
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.env.interface import JointAction
from src.eval.parameter_review import episode_metrics
from src.experiments.research import make_research_env
from scripts.train_clean_research import TrainingWindows
from scripts.train_compact_residual import INPUTS, digest, serial, write_json

STUDY = Path("outputs/rl_discovery/parameter_confirmation_v1")


def collect(env, model, covers, key, label, handle, deadline, *, greedy=False, fixed=None):
    env.reset(seed=key)
    trace, decisions = [], []
    for day in range(1, 29):
        if time.perf_counter() > deadline:
            raise TimeoutError("pricing PPO pilot time cap exceeded")
        decision = None
        if env.pricing_due:
            pobs = env.pricing_observation().copy()
            features = pricing_features(pobs, day, env.base_price)
            state = critic_features(env.get_state(), day)
            prices, logp = model.actor.act(features, greedy=greedy) if fixed is None else (fixed, None)
            decision = dict(obs=pobs, features=features, state=state, raw_state=env.get_state().copy(),
                            action=prices, logp=logp, value=model.value(state))
            decisions.append(serial(decision))
            env.select_price(prices)
        obs = env.replenishment_observation().copy()
        actions = order_action(obs, day, covers)
        _, reward, done, info = env.step(JointAction(None, actions))
        assert done == (day == 28)
        row = serial(dict(label=label, scenario=key, day=day, start=int(env.sim.start_day_index),
                          decision=decision, obs=obs, actions=actions, reward=reward, info=info,
                          demand_seed=int(env.sim.scenario_seed), supplier_seed=int(env.pipeline.scenario_seed)))
        handle.write(json.dumps(row, allow_nan=False) + "\n")
        trace.append(row)
    handle.flush()
    result = episode_metrics(trace)
    result["price_counts"] = np.bincount(np.concatenate([r["action"] for r in decisions]), minlength=5).tolist()
    return dict(decisions=decisions, rewards=[r["reward"] for r in trace]), result, trace


def analyze(rows):
    index = {(r["label"], r["scenario"]): r for r in rows}
    draws = np.random.default_rng(20261024).integers(64, size=(10000, 64))
    result = {}
    for label in ["greedy", "sampled"]:
        for ref in ["baseline", "initial", "static"]:
            for metric in ["reward_day", "fill"]:
                delta = np.array([index[label, k][metric] - index[ref, k][metric]
                                  for k in range(2230000, 2230064)])
                result[f"{label}-{ref}:{metric}"] = dict(mean=float(delta.mean()),
                    ci95=np.quantile(delta[draws].mean(1), [.025, .975]).tolist())
    return result


def run(output, diagnostic):
    out, diagnostic = Path(output), Path(diagnostic)
    assert json.loads((diagnostic / "verification.json").read_text())["status"] == "passed"
    assert json.loads((STUDY / "verification.json").read_text())["confirmed_gate"]
    covers = np.array(json.loads((STUDY / "sku_seed211_final.json").read_text())["mean"])
    static = np.array(json.loads((diagnostic / "selection.json").read_text())["prices"])
    out.mkdir(parents=True, exist_ok=False)
    sources = sorted({str(p) for folder in ["src", "scripts"] for p in Path(folder).rglob("*.py")}
                     | {INPUTS["calibration"], str(STUDY / "sku_seed211_final.json"),
                        str(diagnostic / "selection.json"), str(diagnostic / "verification.json"),
                        "STRUCTURED_PRICING_PPO_PILOT_PROTOCOL.md"})
    hashes = {p: digest(p) for p in sources}
    torch.set_num_threads(1)
    start = time.perf_counter()
    episodes = 0
    try:
        calibration = load_identifiable_training_calibration(INPUTS["calibration"])
        env = TrainingWindows(make_research_env("S2", split="train", start=15, days=28,
                                                params=calibration, pricing_scope="per_sku"), 28)
        env.reset(seed=0)  # Initialize dimensions; no simulator steps.
        model = StructuredPricingPPO(len(env.get_state()), 601)
        initial = StructuredPricingPPO(len(env.get_state()), 601)
        torch.save(model.state(), out / "initial.pt")
        write_json(out / "manifest.json", dict(hashes=hashes, covers=covers, static=static,
            base_prices=env.base_price, state_dim=len(env.get_state()), seed=601,
            updates=64, episodes_per_update=16, test_scoring=False, maximum_seconds=1500,
            planned_episodes=1346))
        with (out / "training.jsonl").open("w") as f:
            history = []
            for u in range(64):
                batch = []
                for e in range(16):
                    data, _, _ = collect(env, model, covers, 2220000 + u * 16 + e,
                                          f"u{u}", f, start + 1500)
                    batch.append(data)
                    episodes += 1
                # Save exact on-policy data and RNG BEFORE any optimizer update.
                torch.save(dict(batch=batch, rng=torch.get_rng_state()), out / f"batch_{u:03d}.pt")
                metrics = model.update(batch)
                torch.save(model.state(), out / f"update_{u + 1:03d}.pt")
                history.append(metrics)
                write_json(out / "progress.json", dict(updates=u + 1, episodes=episodes, metrics=metrics))
                if (u + 1) % 8 == 0:
                    print(f"pricing PPO: {u + 1}/64 updates", flush=True)
        frozen = {name: digest(out / name) for name in ["initial.pt", "update_064.pt"]}
        write_json(out / "assessment_freeze.json", frozen)
        rows, targets = [], {}
        with (out / "assessment.jsonl").open("w") as f:
            for key in range(2230000, 2230064):
                for label in ["baseline", "static", "initial", "greedy", "sampled"]:
                    torch.manual_seed(key + 100000)
                    fixed = np.full(10, 4) if label == "baseline" else static if label == "static" else None
                    actor = initial if label == "initial" else model
                    _, result, trace = collect(env, actor, covers, key, label, f, start + 1500,
                                               greedy=label == "greedy", fixed=fixed)
                    rows.append(result)
                    episodes += 1
                    if key == 2230000 and label in ["greedy", "sampled"]:
                        targets[label] = trace
        with (out / "replays.jsonl").open("w") as f:
            for label in ["greedy", "sampled"]:
                torch.manual_seed(2330000)
                _, _, trace = collect(env, model, covers, 2230000, label, f, start + 1500,
                                       greedy=label == "greedy")
                episodes += 1
                assert trace == targets[label]
        assert episodes == 1346
        assert all(digest(p) == h for p, h in hashes.items())
        assert all(digest(out / p) == h for p, h in frozen.items())
        effects = analyze(rows)
        write_json(out / "summary.json", dict(status="complete", rows=rows, effects=effects,
            history=history, episodes=episodes, days=episodes * 28, seconds=time.perf_counter() - start,
            test_scoring=False, trace_hashes={p.name: digest(p) for p in out.glob("*.jsonl")},
            archive_hashes={p.name: digest(p) for p in out.glob("*.pt")}))
        print(json.dumps(effects), flush=True)
    except Exception as exc:
        write_json(out / "failed_status.json", dict(error=str(exc), complete_episodes=episodes))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--diagnostic", required=True)
    a = p.parse_args()
    run(a.output, a.diagnostic)
