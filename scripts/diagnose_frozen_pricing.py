"""Prospective static pricing diagnostic with frozen successful replenishment."""
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


def candidates():
    policies = {"baseline": np.full(10, 4, dtype=int)}
    policies.update({f"all_{j}": np.full(10, j, dtype=int) for j in range(4)})
    for i in range(10):
        for j in range(4):
            a = np.full(10, 4, dtype=int)
            a[i] = j
            policies[f"sku_{i}_{j}"] = a
    return policies


def select(rows):
    means = {label: np.mean([r["reward_day"] for r in rows if r["label"] == label])
             for label in candidates()}
    return np.array([max([4, 0, 1, 2, 3],
                        key=lambda j: means["baseline" if j == 4 else f"sku_{i}_{j}"])
                     for i in range(10)], dtype=int)


def collect(env, covers, prices, key, label, handle, counters, deadline):
    env.reset(seed=key)
    trace = []
    for day in range(1, 29):
        if time.perf_counter() > deadline:
            raise TimeoutError("pricing diagnostic time cap reached")
        due = env.pricing_due
        pobs = env.pricing_observation().copy() if due else None
        if due:
            env.select_price(prices)
        obs = env.replenishment_observation().copy()
        action = order_action(obs, day, covers)
        _, reward, done, info = env.step(JointAction(None, action))
        assert done == (day == 28)
        row = serial(dict(label=label, scenario=key, start=int(env.sim.start_day_index),
                          day=day, pricing_obs=pobs, price_actions=prices if due else None,
                          obs=obs, actions=action, reward=reward, info=info,
                          demand_seed=int(env.sim.scenario_seed),
                          supplier_seed=int(env.pipeline.scenario_seed)))
        handle.write(json.dumps(row, allow_nan=False) + "\n")
        trace.append(row)
        counters["days"] += 1
    handle.flush()
    counters["episodes"] += 1
    return episode_metrics(trace), trace


def effects(rows):
    index = {(r["label"], r["scenario"]): r for r in rows}
    keys = list(range(2210000, 2210064))
    assert len(index) == len(rows) == 128
    draws = np.random.default_rng(20261023).integers(64, size=(10000, 64))
    result = {}
    for metric in ["reward_day", "fill"]:
        x = np.array([index["selected", k][metric] - index["baseline", k][metric] for k in keys])
        result[metric] = dict(mean=float(x.mean()), ci95=np.quantile(x[draws].mean(1), [.025, .975]).tolist())
    return result


def run(output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    assert json.loads((STUDY / "verification.json").read_text())["confirmed_gate"]
    covers = np.array(json.loads((STUDY / "sku_seed211_final.json").read_text())["mean"])
    sources = sorted({str(p) for folder in ["src", "scripts"] for p in Path(folder).rglob("*.py")}
                     | {INPUTS["calibration"], str(STUDY / "sku_seed211_final.json"),
                        str(STUDY / "verification.json"), "STRUCTURED_MAPPO_EXTENSION_PROTOCOL.md"})
    hashes = {p: digest(p) for p in sources}
    write_json(out / "manifest.json", dict(hashes=hashes, covers=covers, test_scoring=False,
                                          maximum_seconds=900, planned_episodes=850))
    torch.set_num_threads(1)
    start = time.perf_counter()
    counters = dict(episodes=0, days=0)
    try:
        calibration = load_identifiable_training_calibration(INPUTS["calibration"])
        env = TrainingWindows(make_research_env("S2", split="train", start=15, days=28,
                                                params=calibration, pricing_scope="per_sku"), 28)
        discovery = []
        with (out / "discovery.jsonl").open("w") as f:
            for key in range(2200000, 2200016):
                for label, prices in candidates().items():
                    result, _ = collect(env, covers, prices, key, label, f, counters, start + 900)
                    discovery.append(result)
                print(f"pricing discovery: {key - 2200000 + 1}/16", flush=True)
        selected = select(discovery)
        write_json(out / "selection.json", dict(prices=selected, discovery=discovery))
        selected_hash = digest(out / "selection.json")
        rows, targets = [], {}
        with (out / "assessment.jsonl").open("w") as f:
            for key in range(2210000, 2210064):
                for label, prices in dict(baseline=np.full(10, 4), selected=selected).items():
                    result, trace = collect(env, covers, prices, key, label, f, counters, start + 900)
                    rows.append(result)
                    if key == 2210000:
                        targets[label] = trace
        with (out / "replays.jsonl").open("w") as f:
            for label, prices in dict(baseline=np.full(10, 4), selected=selected).items():
                _, trace = collect(env, covers, prices, 2210000, label, f, counters, start + 900)
                assert trace == targets[label]
        assert counters == dict(episodes=850, days=23800)
        assert all(digest(p) == h for p, h in hashes.items())
        assert digest(out / "selection.json") == selected_hash
        result = effects(rows)
        summary = dict(status="complete", prices=selected, effects=result, rows=rows,
                       counters=counters, seconds=time.perf_counter() - start, test_scoring=False,
                       selection_hash=selected_hash,
                       trace_hashes={p.name: digest(p) for p in out.glob("*.jsonl")})
        write_json(out / "summary.json", summary)
        print(json.dumps(serial(dict(prices=selected, effects=result))), flush=True)
    except Exception as exc:
        write_json(out / "failed_status.json", dict(error=str(exc), counters=counters))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
