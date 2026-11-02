"""Reconstruct pricing pilot likelihoods, PPO/Adam updates and paired results."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical
from src.algos.structured_pricing_ppo import StructuredPricingPPO
from scripts.train_compact_residual import digest, write_json
from scripts.verify_parameter_confirmation import episodes, check_episode


def independent_update(model, batch):
    features, actions, logs, states, advantages, targets = [], [], [], [], [], []
    for episode in batch:
        decisions = episode["decisions"]
        rewards = np.array(episode["rewards"]) / 100
        values = [d["value"] for d in decisions]
        adv = np.zeros(4, np.float32)
        tail = 0.
        for week in [3, 2, 1, 0]:
            subtotal = sum(.99 ** k * rewards[7 * week + k] for k in range(7))
            delta = subtotal + .99 ** 7 * (values[week + 1] if week < 3 else 0) - values[week]
            tail = delta + (.99 * .95) ** 7 * tail
            adv[week] = tail
        targets.extend(adv + np.asarray(values, np.float32))
        advantages.extend(adv)
        features.extend([d["features"] for d in decisions])
        actions.extend([d["action"] for d in decisions])
        states.extend([d["state"] for d in decisions])
        logs.extend([d["logp"] for d in decisions])
    x = torch.tensor(np.array(features), dtype=torch.float32)
    a = torch.tensor(np.array(actions), dtype=torch.long)
    old = torch.tensor(logs, dtype=torch.float32)
    adv = torch.tensor(advantages, dtype=torch.float32)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    for _ in range(4):
        dist = Categorical(logits=torch.einsum("...if,ifk->...ik", x, model.actor.weight))
        lp = dist.log_prob(a).sum(-1)
        ratio = torch.exp(lp - old)
        if float((((ratio - 1) - (lp - old)).mean()).detach()) > .02:
            break
        loss = -torch.minimum(ratio * adv, torch.clamp(ratio, .8, 1.2) * adv).mean()
        model.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), .5)
        model.actor_optimizer.step()
    state = torch.tensor(np.array(states), dtype=torch.float32)
    target = torch.tensor(targets, dtype=torch.float32)
    for _ in range(4):
        loss = .5 * (model.critic(state).squeeze(-1) - target).square().mean()
        model.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.critic.parameters(), .5)
        model.critic_optimizer.step()


def verify(output):
    out = Path(output)
    load = lambda p: json.loads((out / p).read_text())
    manifest, summary = load("manifest.json"), load("summary.json")
    assert summary["status"] == "complete"
    torch.set_num_threads(1)
    for name, h in manifest["hashes"].items():
        assert digest(name) == h, name
    for field in ["trace_hashes", "archive_hashes"]:
        for name, h in summary[field].items():
            assert digest(out / name) == h
    for name, h in load("assessment_freeze.json").items():
        assert digest(out / name) == h
    read = lambda p: torch.load(out / p, weights_only=True, map_location="cpu")
    model = StructuredPricingPPO(manifest["state_dim"], 601)
    base = np.array(manifest["base_prices"])
    noise, targets = {}, {}

    def restore(state):
        model.actor.load_state_dict(state["actor"])
        model.critic.load_state_dict(state["critic"])
        model.actor_optimizer.load_state_dict(state["actor_optimizer"])
        model.critic_optimizer.load_state_dict(state["critic_optimizer"])
        torch.set_rng_state(state["rng"])

    def audit(trace, greedy=False, fixed=None):
        scenario = trace[0]["scenario"]
        start = int(np.random.default_rng(np.random.SeedSequence([scenario, 314159])).integers(15, 1859))
        score = check_episode(trace, manifest["covers"], start)
        current = [(r["demand_seed"], r["supplier_seed"]) for r in trace]
        if scenario in noise:
            assert current == noise[scenario]
        else:
            noise[scenario] = current
        decisions = []
        prices = None
        for r in trace:
            d = r["decision"]
            if r["day"] not in [1, 8, 15, 22]:
                assert d is None
            else:
                decisions.append(d)
                x = np.array(d["obs"], np.float32)
                forecast = x[-20:-10]
                features = np.stack([np.ones(10),
                    np.clip(x[:10] / (1 + 7 * forecast), 0, 3),
                    np.log1p(np.maximum(forecast, 0)) / 5,
                    np.clip(x[10:150].reshape(10, 14)[:, -7:].mean(1) / (1 + forecast), 0, 3),
                    x[150:160] / base.astype(np.float32) - 1,
                    np.full(10, (29 - r["day"]) / 28),
                    np.clip(x[-10:] / 500, 0, 1)], -1).astype(np.float32)
                np.testing.assert_array_equal(features, d["features"])
                raw = np.array(d["raw_state"], np.float32)
                state = np.r_[np.sign(raw) * np.log1p(np.abs(raw)) / 5, (29 - r["day"]) / 28].astype(np.float32)
                np.testing.assert_array_equal(state, d["state"])
                with torch.no_grad():
                    value = float(model.critic(torch.tensor(state)).squeeze())
                    assert value == d["value"]
                    if fixed is None:
                        dist = Categorical(logits=torch.einsum("if,ifk->ik", torch.tensor(features), model.actor.weight))
                        action = dist.probs.argmax(-1) if greedy else dist.sample()
                        assert float(dist.log_prob(action).sum()) == d["logp"]
                        prices = action.numpy()
                    else:
                        prices = np.asarray(fixed)
                        assert d["logp"] is None
                np.testing.assert_array_equal(prices, d["action"])
            np.testing.assert_allclose(base * np.array([.8, .9, 1., 1.1, 1.2])[prices],
                                       r["info"]["price"], rtol=0, atol=1e-12)
        score["price_counts"] = np.bincount(np.concatenate([d["action"] for d in decisions]), minlength=5).tolist()
        return score, dict(decisions=decisions, rewards=[r["reward"] for r in trace])

    stream = iter(episodes(out / "training.jsonl"))
    for u in range(64):
        before = read("initial.pt" if u == 0 else f"update_{u:03d}.pt")
        restore(before)
        batch = read(f"batch_{u:03d}.pt")
        assert len(batch["batch"]) == 16
        for e in range(16):
            key, trace = next(stream)
            assert key == (f"u{u}", 2220000 + 16 * u + e)
            _, data = audit(trace)
            assert data == batch["batch"][e]
        assert torch.equal(torch.get_rng_state(), batch["rng"])
        independent_update(model, batch["batch"])
        after = read(f"update_{u + 1:03d}.pt")
        for name in ["actor", "critic"]:
            for k, value in getattr(model, name).state_dict().items():
                assert torch.equal(value, after[name][k]), (u, name, k)
    assert next(stream, None) is None
    results = {}
    for key, trace in episodes(out / "assessment.jsonl"):
        label, scenario = key
        assert key not in results and scenario in range(2230000, 2230064)
        assert label in ["initial", "greedy", "sampled", "baseline", "static"]
        restore(read("initial.pt" if label == "initial" else "update_064.pt"))
        torch.manual_seed(scenario + 100000)
        fixed = np.full(10, 4) if label == "baseline" else manifest["static"] if label == "static" else None
        results[key], _ = audit(trace, label == "greedy", fixed)
        if scenario == 2230000 and label in ["greedy", "sampled"]:
            targets[key] = trace
    assert len(results) == 320
    assert len(summary["rows"]) == len({(r["label"], r["scenario"]) for r in summary["rows"]}) == 320
    for row in summary["rows"]:
        result = results[row["label"], row["scenario"]]
        for metric in ["reward_day", "fill"]:
            assert abs(row[metric] - result[metric]) < 1e-10
        assert row["price_counts"] == result["price_counts"]
    seen = set()
    for key, trace in episodes(out / "replays.jsonl"):
        assert key not in seen and trace == targets[key]
        seen.add(key)
    assert len(seen) == 2
    draws = np.random.default_rng(20261024).integers(0, 64, (10000, 64))
    for label in ["greedy", "sampled"]:
        for ref in ["baseline", "initial", "static"]:
            for metric in ["reward_day", "fill"]:
                delta = np.array([results[label, k][metric] - results[ref, k][metric]
                                  for k in range(2230000, 2230064)])
                expected = summary["effects"][f"{label}-{ref}:{metric}"]
                assert abs(expected["mean"] - delta.mean()) < 1e-10
                np.testing.assert_allclose(np.quantile([delta[d].mean() for d in draws], [.025, .975]),
                                           expected["ci95"], atol=1e-10, rtol=0)
    assert summary["episodes"] == 1346 and summary["days"] == 37688
    assert summary["seconds"] <= 1500 and not summary["test_scoring"]
    write_json(out / "verification.json", dict(status="passed", training_episodes=1024,
        assessment_episodes=320, replays=2, independent_ppo_updates=64,
        actions_returns_features_rng_intervals=True, new_simulator_days=0,
        verifier_sha256=digest(__file__)))
    print("Structured pricing PPO pilot verified.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    verify(p.parse_args().output)
