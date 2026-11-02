"""Archive-only checks independent of pricing diagnostic selection/runner."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.config.sku_config import load_skus
from scripts.train_compact_residual import digest, write_json
from scripts.verify_parameter_confirmation import episodes, check_episode


def verify(output):
    out = Path(output)
    load = lambda name: json.loads((out / name).read_text())
    manifest, summary, selection = [load(n) for n in ["manifest.json", "summary.json", "selection.json"]]
    assert summary["status"] == "complete"
    for p, h in manifest["hashes"].items():
        assert digest(p) == h, p
    for name, h in summary["trace_hashes"].items():
        assert digest(out / name) == h
    assert digest(out / "selection.json") == summary["selection_hash"]
    calibration = next(p for p in manifest["hashes"] if p.endswith("train_only_nb_identifiable_v2.json"))
    order = json.loads(Path(calibration).read_text())["series_order"]
    lookup = {s.series_id: s.base_price for s in load_skus()}
    base = np.array([lookup[s] for s in order])
    noise = {}

    def audit(trace, prices):
        key = trace[0]["scenario"]
        start = int(np.random.default_rng(np.random.SeedSequence([key, 314159])).integers(15, 1859))
        result = check_episode(trace, manifest["covers"], start)
        current = [(r["demand_seed"], r["supplier_seed"]) for r in trace]
        if key in noise:
            assert noise[key] == current
        else:
            noise[key] = current
        for row in trace:
            if row["day"] in [1, 8, 15, 22]:
                np.testing.assert_array_equal(prices, row["price_actions"])
                assert row["pricing_obs"] is not None
            else:
                assert row["price_actions"] is None and row["pricing_obs"] is None
            np.testing.assert_allclose(base * np.array([.8, .9, 1., 1.1, 1.2])[prices],
                                       row["info"]["price"], rtol=0, atol=1e-12)
        return result

    labels = ["baseline"] + [f"all_{j}" for j in range(4)] + [
        f"sku_{i}_{j}" for i in range(10) for j in range(4)]
    discovery = {}
    for key, trace in episodes(out / "discovery.jsonl"):
        label, scenario = key
        assert label in labels and scenario in range(2200000, 2200016) and key not in discovery
        prices = np.full(10, 4)
        if label.startswith("all_"):
            prices[:] = int(label.split("_")[1])
        if label.startswith("sku_"):
            _, i, j = label.split("_")
            prices[int(i)] = int(j)
        discovery[key] = audit(trace, prices)
    assert len(discovery) == 720
    means = {l: np.mean([discovery[l, k]["reward_day"] for k in range(2200000, 2200016)]) for l in labels}
    selected = []
    for i in range(10):
        best, best_score = 4, means["baseline"]
        for j in range(4):
            if means[f"sku_{i}_{j}"] > best_score:
                best, best_score = j, means[f"sku_{i}_{j}"]
        selected.append(best)
    np.testing.assert_array_equal(selected, selection["prices"])
    np.testing.assert_array_equal(selected, summary["prices"])
    for row in selection["discovery"]:
        for metric in ["reward_day", "fill"]:
            assert abs(row[metric] - discovery[row["label"], row["scenario"]][metric]) < 1e-10
    assert len(selection["discovery"]) == 720
    assessment, targets = {}, {}
    for key, trace in episodes(out / "assessment.jsonl"):
        label, scenario = key
        assert label in ["baseline", "selected"] and scenario in range(2210000, 2210064)
        assert key not in assessment
        assessment[key] = audit(trace, np.array(selected) if label == "selected" else np.full(10, 4))
        if scenario == 2210000:
            targets[key] = trace
    assert len(assessment) == 128
    assert len(summary["rows"]) == len({(r["label"], r["scenario"]) for r in summary["rows"]}) == 128
    for row in summary["rows"]:
        for metric in ["reward_day", "fill"]:
            assert abs(row[metric] - assessment[row["label"], row["scenario"]][metric]) < 1e-10
    seen = set()
    for key, trace in episodes(out / "replays.jsonl"):
        assert key not in seen and trace == targets[key]
        seen.add(key)
    assert len(seen) == 2
    draws = np.random.default_rng(20261023).integers(0, 64, (10000, 64))
    for metric in ["reward_day", "fill"]:
        delta = np.array([assessment["selected", k][metric] - assessment["baseline", k][metric]
                          for k in range(2210000, 2210064)])
        expected = summary["effects"][metric]
        assert abs(expected["mean"] - delta.mean()) < 1e-10
        np.testing.assert_allclose(np.quantile([delta[d].mean() for d in draws], [.025, .975]),
                                   expected["ci95"], rtol=0, atol=1e-10)
    assert summary["counters"] == dict(episodes=850, days=23800)
    assert summary["seconds"] <= 900 and not summary["test_scoring"]
    write_json(out / "verification.json", dict(status="passed", episodes=850, exact_replays=2,
               actions_selection_returns_intervals=True, paired_exogenous_seeds=True,
               new_simulator_days=0, verifier_sha256=digest(__file__)))
    print("Frozen pricing diagnostic verified.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    verify(p.parse_args().output)
