"""Independent archived action/economic reconstruction for factorial diagnostic."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.config.sku_config import load_skus
from scripts.train_compact_residual import digest, write_json
from scripts.verify_parameter_confirmation import episodes, check_episode


def verify(output):
    out = Path(output)
    manifest = json.loads((out / "manifest.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "complete"
    for p, h in manifest["hashes"].items():
        assert digest(p) == h, p
    for p, h in summary["trace_hashes"].items():
        assert digest(out / p) == h
    calibration = next(p for p in manifest["hashes"] if p.endswith("train_only_nb_identifiable_v2.json"))
    order = json.loads(Path(calibration).read_text())["series_order"]
    lookup = {s.series_id: s.base_price for s in load_skus()}
    base = np.array([lookup[s] for s in order])
    covers = np.array(manifest["covers"])
    results, noises, targets = {}, {}, {}
    for key, trace in episodes(out / "traces.jsonl"):
        label, scenario = key
        assert key not in results and scenario in range(2240000, 2240064)
        assert label in [f"O{o}P{p}" for o in range(3) for p in range(3)]
        o, p = int(label[1]), int(label[3])
        start = int(np.random.default_rng(np.random.SeedSequence([scenario, 314159])).integers(15, 1859))
        result = check_episode(trace, start=start)
        counts = np.zeros(5, int)
        for r in trace:
            day = r["day"]
            x = np.asarray(r["obs"], np.float32)
            c = np.minimum(covers, max(0, 25 - day)) if o == 1 else covers
            pipe = x[10:80].reshape(10, 7)
            a = np.ceil(np.maximum(np.asarray(c, np.float32).clip(0, 20) * x[-10:]
                - (x[:10] + pipe.sum(1)), 0) / 5).clip(0, 40).astype(int)
            if o == 2 and day >= 22:
                a[:] = 0
            np.testing.assert_array_equal(a, r["actions"])
            if day in [1, 8, 15, 22]:
                obs = np.array(r["pricing_obs"], np.float32)
                pa = np.full(10, 4, dtype=int)
                if p:
                    excess = obs[:10] > 1.5 * min(7, 29 - day) * obs[-20:-10]
                    pa[excess] = 3 if p == 1 else 0
                np.testing.assert_array_equal(pa, r["price_actions"])
                counts += np.bincount(pa, minlength=5)
                current_prices = base * np.array([.8, .9, 1., 1.1, 1.2])[pa]
            else:
                assert r["price_actions"] is None and r["pricing_obs"] is None
            np.testing.assert_allclose(current_prices, r["info"]["price"], rtol=0, atol=1e-12)
        noise = [(r["demand_seed"], r["supplier_seed"]) for r in trace]
        if scenario in noises:
            assert noises[scenario] == noise
        else:
            noises[scenario] = noise
        result["price_counts"] = counts.tolist()
        results[key] = result
        if scenario == 2240000 and label in ["O0P0", "O1P1"]:
            targets[key] = trace
    assert len(results) == 576
    assert len(summary["rows"]) == len({(r["label"], r["scenario"]) for r in summary["rows"]}) == 576
    for r in summary["rows"]:
        expected = results[r["label"], r["scenario"]]
        for metric in ["reward_day", "fill"]:
            assert abs(expected[metric] - r[metric]) < 1e-10
        assert expected["price_counts"] == r["price_counts"]
    seen = set()
    for key, trace in episodes(out / "replays.jsonl"):
        assert key not in seen and trace == targets[key]
        seen.add(key)
    assert len(seen) == 2
    draws = np.random.default_rng(20261025).integers(0, 64, (10000, 64))
    for name, expected in summary["analysis"]["effects"].items():
        pair, metric = name.split(":")
        label, ref = pair.split("-")
        delta = np.array([results[label, k][metric] - results[ref, k][metric] for k in range(2240000, 2240064)])
        assert abs(expected["mean"] - delta.mean()) < 1e-10
        np.testing.assert_allclose(np.quantile([delta[d].mean() for d in draws], [.025, .975]),
                                   expected["ci95"], rtol=0, atol=1e-10)
    for label, metrics in summary["analysis"]["means"].items():
        for metric, mean in metrics.items():
            assert abs(mean - np.mean([results[label, k][metric] for k in range(2240000, 2240064)])) < 1e-10
    assert summary["episodes"] == 578 and summary["days"] == 16184
    assert summary["seconds"] <= 600 and not summary["test_scoring"]
    write_json(out / "verification.json", dict(status="passed", episodes=578, exact_replays=2,
        independent_actions_returns_intervals=True, paired_exogenous_seeds=True, new_simulator_days=0,
        verifier_sha256=digest(__file__)))
    print("Coupled opportunity diagnostic verified.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    verify(p.parse_args().output)
