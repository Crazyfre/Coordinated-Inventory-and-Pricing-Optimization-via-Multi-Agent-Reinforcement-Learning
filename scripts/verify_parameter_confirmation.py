"""Streaming independent reconstruction of frozen five-seed parameter RL."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from scripts.train_compact_residual import digest, write_json


def episodes(path):
    current = []; key = None
    with path.open() as handle:
        for line in handle:
            row = json.loads(line); newkey = (row['label'], row['scenario'])
            if key is not None and newkey != key:
                yield key, current; current = []
            key = newkey; current.append(row)
    if current:
        yield key, current


def check_episode(rows, parameters=None, start=None):
    assert [r['day'] for r in rows] == list(range(1, 29))
    total, discounted, sold, demand = 0., 0., 0., 0.
    for r in rows:
        if start is not None:
            assert r['start'] == start
        terms = r['info']['reward_terms']
        reward = terms['revenue']-sum(terms[k] for k in ['ordering_cost', 'holding_cost', 'stockout_cost', 'price_change_cost'])
        assert abs(reward-r['reward']) < 1e-8 and r['info']['total_on_hand'] <= 500+1e-8
        if parameters is not None:
            x, p = np.asarray(r['obs'], np.float32), np.asarray(parameters, np.float32)
            pipeline = x[10:80].reshape(10, 7); cover = p[:10].copy()
            if len(p) == 21:
                late = pipeline[:, 3:].sum(1)/(1+pipeline.sum(1))
                cover += p[10:20]*late+p[20]*np.float32((29-r['day'])/28-.5)
            action = np.ceil(np.maximum(cover.clip(0, 20)*x[-10:]-(x[:10]+pipeline.sum(1)), 0)/5).clip(0, 40).astype(int)
            np.testing.assert_array_equal(action, r['actions'])
        total += r['reward']; discounted += .99**(r['day']-1)*r['reward']
        sold += np.asarray(r['info']['sales']).sum(); demand += np.asarray(r['info']['demand']).sum()
    return dict(reward_day=total/28, discounted_day=discounted/28, fill=float(sold/demand) if demand else 1.)


def verify(output):
    out = Path(output); load = lambda p: json.loads(p.read_text())
    manifest, summary = load(out/'manifest.json'), load(out/'summary.json')
    for p, h in manifest['hashes'].items():
        assert digest(p) == h
    for p, h in summary['trace_hashes'].items():
        assert digest(out/p) == h
    seeds = [211, 222, 233, 244, 255]; arm = manifest['arm']; dimension = 10 if arm == 'sku' else 21
    sigma = np.r_[np.full(10, .75), np.full(dimension-10, .5)]
    initial = np.r_[np.full(10, 7.5), np.zeros(dimension-10)]
    parameters, expected_scores, finals = {}, {}, {}
    for j, seed in enumerate(seeds):
        rng = np.random.default_rng(seed); theta = initial.copy(); m = np.zeros(dimension); v = m.copy()
        history = load(out/f'{arm}_seed{seed}_history.json'); assert len(history) == 64
        for u, record in enumerate(history):
            e = rng.normal(size=(8, dimension)); np.testing.assert_array_equal(e, record['epsilon'])
            np.testing.assert_array_equal(theta, record['mean_before'])
            diffs = []
            for pair, rows in enumerate(record['records']):
                key = 1200000+j*10000+u*8+pair; assert rows['key'] == key
                for sign, direction in [('plus', 1), ('minus', -1)]:
                    label = f'{arm}_{seed}_u{u}_p{pair}_{sign}'
                    parameters[label, key] = theta+direction*sigma*e[pair]
                    expected_scores[label, key] = rows[sign]
                diffs.append(rows['plus']['discounted_day']-rows['minus']['discounted_day'])
            np.testing.assert_array_equal(diffs, record['differences'])
            gradient = np.mean(np.array(diffs)[:, None]*e/(2*sigma), axis=0)
            np.testing.assert_array_equal(gradient, record['gradient'])
            m = .9*m+.1*gradient; v = .999*v+.001*gradient**2
            theta += .1*(m/(1-.9**(u+1)))/(np.sqrt(v/(1-.999**(u+1)))+1e-8)
            theta[:10] = theta[:10].clip(0, 16); theta[10:] = theta[10:].clip(-8, 8)
            np.testing.assert_array_equal(theta, record['state']['mean'])
        finals[seed] = theta.copy(); path = out/f'{arm}_seed{seed}_final.json'
        np.testing.assert_array_equal(theta, load(path)['mean'])
        assert digest(path) == load(out/'frozen_checkpoints.json')[str(seed)]
    count = 0; pair_demand = None
    for key, rows in episodes(out/'training_traces.jsonl'):
        start = int(np.random.default_rng(np.random.SeedSequence([key[1], 314159])).integers(15, 1859))
        result = check_episode(rows, parameters[key], start)
        for metric, value in result.items():
            assert abs(value-expected_scores[key][metric]) < 1e-12
        demand = [r['info']['demand'] for r in rows]
        if key[0].endswith('_plus'):
            pair_demand = (key[1], demand)
        else:
            assert pair_demand == (key[1], demand)
        count += 1
    assert count == 5120
    print('Verified five seeds,320 independent score-function/Adam updates and5120 training episodes.', flush=True)
    recomputed = {}; reference_noise = {}; replay_targets = {}
    for key, rows in episodes(out/'assessment_traces.jsonl'):
        label, scenario = key
        p = None
        if label.startswith('learned_'):
            p = finals[int(label.split('_')[1])]
        elif label.startswith('initial_'):
            seed = int(label.split('_')[1]); rng = np.random.default_rng(np.random.SeedSequence([seed, scenario, 707]))
            p = initial+sigma*rng.normal(size=dimension)
        elif label == 'full7_5':
            p = np.full(10, 7.5)
        recomputed[key] = check_episode(rows, p, 1886)
        noise = [(r['info']['demand'], r['info']['price']) for r in rows]
        if scenario in reference_noise:
            assert reference_noise[scenario] == noise
        else:
            reference_noise[scenario] = noise
        if scenario == 1300000 and label.startswith(('learned_', 'initial_')):
            replay_targets[label] = rows
    assert len(recomputed) == 1792 and len(reference_noise) == 128
    for row in summary['rows']:
        for metric, value in recomputed[row['label'], row['scenario']].items():
            assert abs(value-row[metric]) < 1e-12
    replay_count = 0
    for key, rows in episodes(out/'replay_traces.jsonl'):
        assert rows == replay_targets[key[0]]; replay_count += 1
    assert replay_count == 10
    names = [(ref, metric) for ref in ['full7_5', 'initial'] for metric in ['reward_day', 'fill']]
    arrays = [np.array([[recomputed[f'learned_{s}', k][metric]-recomputed[(f'initial_{s}' if ref == 'initial' else ref), k][metric]
                        for k in range(1300000, 1300128)] for s in seeds]) for ref, metric in names]
    rng = np.random.default_rng(20261019); draws = np.zeros((10000, 4))
    for b in range(10000):
        r = rng.integers(0, 5, 5); c = rng.integers(0, 128, 128)
        for i, values in enumerate(arrays):
            draws[b, i] = values[np.ix_(r, c)].mean()
    passed = True
    for i, ((ref, metric), values) in enumerate(zip(names, arrays)):
        ep = summary['analysis']['endpoints'][f'{ref}_{metric}']; bounds = np.quantile(draws[:, i], [.00625, .99375])
        np.testing.assert_allclose(bounds, ep['ci9875'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(values.mean(1), ep['seed_means'], atol=1e-12, rtol=0)
        assert abs(values.mean()-ep['mean']) < 1e-12
        for row, interval in zip(values, ep['conditional_seed_ci95']):
            w = t.ppf(.975, 127)*row.std(ddof=1)/np.sqrt(128)
            np.testing.assert_allclose([row.mean()-w, row.mean()+w], interval, atol=1e-12, rtol=0)
        gate = bool(bounds[0] > 0 if metric == 'reward_day' else bounds[0] >= -.005)
        assert gate == ep['passed']; passed &= gate
        if metric == 'reward_day':
            passed &= bool(np.all(values.mean(1) > 0))
    assert bool(passed) == summary['analysis']['learning_gate']
    assert summary['counters'] == {'episodes': 6922, 'days': 193816}
    assert summary['training_seconds'] <= 3600 and summary['assessment_seconds'] <= 1500
    write_json(out/'verification.json', dict(status='passed', verifier_sha256=digest(__file__),
              reconstructed_updates=320, checked_training_episodes=5120, checked_assessment_episodes=1792,
              exact_checkpoint_replays=10, independent_actions_rewards_gradients_intervals=True,
              confirmed_gate=bool(passed), new_simulator_days=0, test_scoring=False))
    print('Verified confirmation actions,paired noise,ten replays,independent bootstrap and gate.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); verify(p.parse_args().output)
