"""Independent daily GAE/latent PPO replay, physical actions and result audit."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Normal
from src.algos.structured_ordering_ppo import StructuredOrderingPPO
from scripts.train_compact_residual import digest, write_json
from scripts.verify_parameter_confirmation import episodes, check_episode


def independent_update(model, batch):
    features, actions, logs, states, advantages, targets = [], [], [], [], [], []
    for episode in batch:
        rows = episode['decisions']; values = [r['value'] for r in rows]
        rewards = np.asarray(episode['rewards']) / 100
        adv = np.zeros(28, np.float32); tail = 0.
        for day in reversed(range(28)):
            delta = rewards[day] + .99 * (values[day + 1] if day < 27 else 0) - values[day]
            tail = delta + (.99 * .95) * tail; adv[day] = tail
        targets.extend(adv + np.asarray(values, np.float32)); advantages.extend(adv)
        features.extend([r['features'] for r in rows]); actions.extend([r['action'] for r in rows])
        logs.extend([r['logp'] for r in rows]); states.extend([r['state'] for r in rows])
    x = torch.tensor(np.array(features), dtype=torch.float32)
    z = torch.tensor(np.array(actions), dtype=torch.float32)
    old = torch.tensor(logs, dtype=torch.float32)
    adv = torch.tensor(advantages, dtype=torch.float32)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    for _ in range(4):
        dist = Normal(model.actor.covers + (x * model.actor.weight).sum(-1), .75)
        lp = dist.log_prob(z).sum(-1); ratio = torch.exp(lp - old)
        if float((((ratio - 1) - (lp - old)).mean()).detach()) > .02:
            break
        loss = -torch.minimum(ratio * adv, ratio.clamp(.8, 1.2) * adv).mean()
        model.actor_optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), .5); model.actor_optimizer.step()
    state = torch.tensor(np.array(states), dtype=torch.float32)
    target = torch.tensor(targets, dtype=torch.float32)
    for _ in range(4):
        loss = .5 * (model.critic(state).squeeze(-1) - target).square().mean()
        model.critic_optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.critic.parameters(), .5); model.critic_optimizer.step()


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal_state(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal_state(x, y)
    else:
        assert a == b


def verify(output):
    out = Path(output); load = lambda p: json.loads((out / p).read_text())
    manifest, summary = load('manifest.json'), load('summary.json')
    assert summary['status'] == 'complete'; torch.set_num_threads(1)
    for p, h in manifest['hashes'].items(): assert digest(p) == h, p
    for field in ['trace_hashes', 'archive_hashes']:
        for p, h in summary[field].items(): assert digest(out / p) == h
    for p, h in load('assessment_freeze.json').items(): assert digest(out / p) == h
    read = lambda p: torch.load(out / p, weights_only=True, map_location='cpu')
    model = StructuredOrderingPPO(manifest['state_dim'], manifest['covers'], 701)
    noises = {}; initial = read('initial.pt')
    equal_state(model.state(), initial)

    def restore(state):
        model.actor.load_state_dict(state['actor']); model.critic.load_state_dict(state['critic'])
        model.actor_optimizer.load_state_dict(state['actor_optimizer'])
        model.critic_optimizer.load_state_dict(state['critic_optimizer']); torch.set_rng_state(state['rng'])

    def audit(trace, greedy=False, baseline=False):
        scenario = trace[0]['scenario']
        start = int(np.random.default_rng(np.random.SeedSequence([scenario, 314159])).integers(15, 1859))
        score = check_episode(trace, start=start)
        noise = [(r['demand_seed'], r['supplier_seed']) for r in trace]
        if scenario in noises: assert noises[scenario] == noise
        else: noises[scenario] = noise
        for r in trace:
            d = r['decision']; x = np.array(r['obs'], np.float32); day = r['day']
            pipe = x[10:80].reshape(10, 7); forecast = x[-10:]
            f = np.stack([np.ones(10), np.clip(x[:10] / (1 + forecast) / 20, 0, 2),
                np.clip(pipe.sum(1) / (1 + forecast) / 20, 0, 2),
                pipe[:, 3:].sum(1) / (1 + pipe.sum(1)), np.log1p(np.maximum(forecast, 0)) / 5,
                np.full(10, (29 - day) / 28), np.clip(x[-20:-10] / 500, 0, 1)], -1).astype(np.float32)
            np.testing.assert_array_equal(f, d['features'])
            raw = np.array(d['raw_state'], np.float32)
            np.testing.assert_array_equal(raw[:260], x)
            state = np.r_[np.sign(raw) * np.log1p(np.abs(raw)) / 5, (29 - day) / 28].astype(np.float32)
            np.testing.assert_array_equal(state, d['state'])
            with torch.no_grad():
                assert float(model.critic(torch.tensor(state)).squeeze()) == d['value']
                if baseline:
                    latent = np.array(manifest['covers'], np.float32); assert d['logp'] is None
                else:
                    dist = Normal(model.actor.covers + (torch.tensor(f) * model.actor.weight).sum(-1), .75)
                    z = dist.mean if greedy else dist.sample()
                    assert float(dist.log_prob(z).sum()) == d['logp']; latent = z.numpy()
            np.testing.assert_array_equal(latent, d['action'])
            action = np.ceil(np.maximum(latent.clip(0, 20) * forecast - (x[:10] + pipe.sum(1)), 0) / 5).clip(0, 40).astype(int)
            np.testing.assert_array_equal(action, r['actions'])
            np.testing.assert_allclose(np.array(manifest['base_prices']) * 1.2, r['info']['price'], atol=1e-12, rtol=0)
        return score, dict(decisions=[r['decision'] for r in trace], rewards=[r['reward'] for r in trace])

    stream = iter(episodes(out / 'training.jsonl'))
    for u in range(64):
        restore(read('initial.pt' if u == 0 else f'update_{u:03d}.pt'))
        batch = read(f'batch_{u:03d}.pt'); assert len(batch['batch']) == 16
        for e in range(16):
            key, trace = next(stream); assert key == (f'u{u}', 2250000 + u * 16 + e)
            _, data = audit(trace); assert data == batch['batch'][e]
        assert torch.equal(torch.get_rng_state(), batch['rng'])
        independent_update(model, batch['batch']); equal_state(model.state(), read(f'update_{u + 1:03d}.pt'))
    assert next(stream, None) is None
    results, targets = {}, {}
    for key, trace in episodes(out / 'assessment.jsonl'):
        label, scenario = key
        assert key not in results and scenario in range(2260000, 2260064)
        assert label in ['initial', 'greedy', 'sampled', 'baseline']
        restore(read('initial.pt' if label == 'initial' else 'update_064.pt'))
        torch.manual_seed(scenario + 100000)
        results[key], _ = audit(trace, label == 'greedy', label == 'baseline')
        if scenario == 2260000 and label in ['greedy', 'sampled']: targets[key] = trace
    assert len(results) == 256
    assert len(summary['rows']) == len({(r['label'], r['scenario']) for r in summary['rows']}) == 256
    for r in summary['rows']:
        for metric in ['reward_day', 'fill']:
            assert abs(r[metric] - results[r['label'], r['scenario']][metric]) < 1e-10
    seen = set()
    for key, trace in episodes(out / 'replays.jsonl'):
        assert key not in seen and trace == targets[key]; seen.add(key)
    assert len(seen) == 2
    draws = np.random.default_rng(20261026).integers(64, size=(10000, 64))
    for label in ['greedy', 'sampled']:
        for ref in ['baseline', 'initial']:
            for metric in ['reward_day', 'fill']:
                delta = np.array([results[label, k][metric] - results[ref, k][metric] for k in range(2260000, 2260064)])
                expected = summary['effects'][f'{label}-{ref}:{metric}']
                assert abs(expected['mean'] - delta.mean()) < 1e-10
                np.testing.assert_allclose(np.quantile([delta[d].mean() for d in draws], [.025, .975]), expected['ci95'], atol=1e-10, rtol=0)
    assert summary['episodes'] == 1282 and summary['days'] == 35896
    assert summary['seconds'] <= 1800 and not summary['test_scoring']
    write_json(out / 'verification.json', dict(status='passed', training_episodes=1024,
        assessment_episodes=256, exact_replays=2, independent_ppo_updates=64,
        actions_features_rng_optimizer_returns_intervals=True, new_simulator_days=0, verifier_sha256=digest(__file__)))
    print('Structured ordering PPO pilot verified.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); verify(p.parse_args().output)
