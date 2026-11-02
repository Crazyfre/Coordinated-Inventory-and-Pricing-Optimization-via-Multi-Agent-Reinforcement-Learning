"""Small prospectively frozen factorial test, not an RL training experiment."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.algos.parameter_policy_gradient import order_action
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.env.interface import JointAction
from src.eval.parameter_review import episode_metrics
from src.experiments.research import make_research_env
from scripts.train_clean_research import TrainingWindows
from scripts.train_compact_residual import INPUTS, digest, serial, write_json

STUDY = Path("outputs/rl_discovery/parameter_confirmation_v1")


def actions(obs, day, covers, ordering):
    if ordering == 2 and day >= 22:
        return np.zeros(10, dtype=int)
    p = np.minimum(covers, max(0, 29 - day - 4)) if ordering == 1 else covers
    return order_action(obs, day, p)


def prices(obs, day, pricing):
    result = np.full(10, 4, dtype=int)
    if pricing:
        result[obs[:10] > 1.5 * min(7, 29 - day) * obs[-20:-10]] = 3 if pricing == 1 else 0
    return result


def collect(env, covers, key, ordering, pricing, handle, deadline):
    env.reset(seed=key)
    trace = []
    for day in range(1, 29):
        if time.perf_counter() > deadline:
            raise TimeoutError("coupled diagnostic time cap")
        pobs = env.pricing_observation().copy() if env.pricing_due else None
        pa = prices(pobs, day, pricing) if pobs is not None else None
        if pa is not None:
            env.select_price(pa)
        obs = env.replenishment_observation().copy()
        action = actions(obs, day, covers, ordering)
        _, reward, done, info = env.step(JointAction(None, action))
        assert done == (day == 28)
        row = serial(dict(label=f"O{ordering}P{pricing}", scenario=key, day=day,
            start=int(env.sim.start_day_index), obs=obs, actions=action, pricing_obs=pobs,
            price_actions=pa, reward=reward, info=info, demand_seed=int(env.sim.scenario_seed),
            supplier_seed=int(env.pipeline.scenario_seed)))
        handle.write(json.dumps(row, allow_nan=False) + "\n")
        trace.append(row)
    handle.flush()
    result = episode_metrics(trace)
    result["price_counts"] = np.bincount(np.concatenate([r["price_actions"] for r in trace
        if r["price_actions"] is not None]), minlength=5).tolist()
    return result, trace


def analyze(rows):
    index = {(r["label"], r["scenario"]): r for r in rows}
    assert len(index) == len(rows) == 576
    draws = np.random.default_rng(20261025).integers(64, size=(10000, 64))
    means, effects = {}, {}
    for o in range(3):
        for p in range(3):
            label = f"O{o}P{p}"
            means[label] = {m: float(np.mean([index[label, k][m] for k in range(2240000, 2240064)]))
                            for m in ["reward_day", "fill"]}
            for ref in sorted({"O0P0", f"O{o}P0"}):
                for metric in ["reward_day", "fill"]:
                    delta = np.array([index[label, k][metric] - index[ref, k][metric]
                                      for k in range(2240000, 2240064)])
                    effects[f"{label}-{ref}:{metric}"] = dict(mean=float(delta.mean()),
                        ci95=np.quantile(delta[draws].mean(1), [.025, .975]).tolist())
    return dict(means=means, effects=effects)


def run(output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    covers = np.array(json.loads((STUDY / "sku_seed211_final.json").read_text())["mean"])
    assert json.loads((STUDY / "verification.json").read_text())["confirmed_gate"]
    sources = sorted({str(p) for folder in ["src", "scripts"] for p in Path(folder).rglob("*.py")}
        | {INPUTS["calibration"], str(STUDY / "sku_seed211_final.json"), "COUPLED_OPPORTUNITY_PROTOCOL.md"})
    hashes = {p: digest(p) for p in sources}
    write_json(out / "manifest.json", dict(hashes=hashes, covers=covers, maximum_seconds=600))
    torch.set_num_threads(1)
    started = time.perf_counter()
    count = 0
    try:
        calibration = load_identifiable_training_calibration(INPUTS["calibration"])
        env = TrainingWindows(make_research_env("S2", split="train", start=15, days=28,
                                                params=calibration, pricing_scope="per_sku"), 28)
        rows, targets = [], {}
        with (out / "traces.jsonl").open("w") as f:
            for key in range(2240000, 2240064):
                for o in range(3):
                    for p in range(3):
                        result, trace = collect(env, covers, key, o, p, f, started + 600)
                        count += 1
                        rows.append(result)
                        if key == 2240000 and (o, p) in [(0, 0), (1, 1)]:
                            targets[o, p] = trace
                if (key - 2240000 + 1) % 8 == 0:
                    print(f"coupled diagnostic: {key - 2240000 + 1}/64 scenarios", flush=True)
        with (out / "replays.jsonl").open("w") as f:
            for o, p in [(0, 0), (1, 1)]:
                _, trace = collect(env, covers, 2240000, o, p, f, started + 600)
                count += 1
                assert trace == targets[o, p]
        assert count == 578 and all(digest(p) == h for p, h in hashes.items())
        result = analyze(rows)
        write_json(out / "summary.json", dict(status="complete", rows=rows, analysis=result,
            episodes=count, days=count * 28, seconds=time.perf_counter() - started,
            test_scoring=False, trace_hashes={p.name: digest(p) for p in out.glob("*.jsonl")}))
        print(json.dumps(result["means"]), flush=True)
    except Exception as exc:
        write_json(out / "failed_status.json", dict(error=str(exc), complete_episodes=count))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    run(p.parse_args().output)
