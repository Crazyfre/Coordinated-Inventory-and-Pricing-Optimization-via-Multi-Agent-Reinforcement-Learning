"""Reconstruct frozen compact study from archived traces; no simulator episodes."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t
from src.algos.compact_residual import CompactResidualPPO
from scripts.train_compact_residual import digest, write_json


def load(path):
    return json.loads(Path(path).read_text())


def check_reward(row):
    terms = row['info']['reward_terms']
    total = terms['revenue']-sum(terms[k] for k in ('ordering_cost', 'holding_cost', 'stockout_cost', 'price_change_cost'))
    assert abs(total-row['reward']) < 1e-8
    assert row['info']['total_on_hand'] <= 500+1e-8


def from_checkpoint(path):
    data = torch.load(path, weights_only=True)
    s = data['spec']
    model = CompactResidualPPO(s['state_dim'], s['coefficients'], s['identity'], s['provenance'], s['seed'])
    return model, data


def verify(output):
    out = Path(output); torch.set_num_threads(1)
    manifest, summary = load(out/'manifest.json'), load(out/'summary.json')
    assert summary['status'] == 'complete'
    for path, expected in manifest['hashes'].items():
        assert digest(path) == expected, path
    for path, expected in load(out/'frozen_checkpoints.json').items():
        assert digest(path) == expected, path
    training = load(out/'training'/'summary.json')
    assert training['completed_seeds'] == [11, 22, 33] and training['episodes'] == 1536
    assert training['simulated_days'] == 43008 and training['seconds'] <= 1800
    # Recompute every training action and batch update; no new environment steps.
    for seed, first_key in [(11, 1000000), (22, 1001000), (33, 1002000)]:
        directory = out/'training'/f'seed{seed}'
        assert digest(directory/'traces.jsonl') == summary['training_trace_hashes'][str(seed)]
        model, initial = from_checkpoint(directory/'initial.pt')
        for key, value in model.actor.state_dict().items():
            torch.testing.assert_close(value, initial['actor'][key], rtol=0, atol=0)
        for key, value in model.critic.state_dict().items():
            torch.testing.assert_close(value, initial['critic'][key], rtol=0, atol=0)
        model.load(directory/'initial.pt')
        history = load(directory/'history.json'); assert len(history) == 64
        with (directory/'traces.jsonl').open() as handle:
            for update in range(64):
                episodes = []
                for j in range(8):
                    key = first_key+update*8+j
                    start = int(np.random.default_rng(np.random.SeedSequence([key, 314159])).integers(15, 1859))
                    episode = []
                    for day in range(1, 29):
                        row = json.loads(next(handle))
                        assert (row['scenario'], row['day'], row['update'], row['start']) == (key, day, update, start)
                        check_reward(row)
                        for field in ['obs', 'state', 'action']:
                            row[field] = np.asarray(row[field])
                        action, lp = model.actor.act(row['obs'], day, generator=model.generator)
                        np.testing.assert_array_equal(action, row['action'])
                        assert lp == row['log_prob'] and model.value(row['state']) == row['value']
                        if day >= 27:
                            np.testing.assert_array_equal(action, row['default'])
                        episode.append(row)
                    episodes.append(episode)
                result = model.update(episodes)
                for key in result:
                    assert result[key] == history[update][key], (seed, update, key)
                if update+1 in [16, 32, 48, 64]:
                    name = 'final.pt' if update == 63 else f'episode{8*(update+1)}.pt'
                    saved = torch.load(directory/name, weights_only=True)
                    for key, value in model.actor.state_dict().items():
                        torch.testing.assert_close(value, saved['actor'][key], rtol=0, atol=0)
                    for key, value in model.critic.state_dict().items():
                        torch.testing.assert_close(value, saved['critic'][key], rtol=0, atol=0)
                    torch.testing.assert_close(model.generator.get_state(), saved['action_rng'], rtol=0, atol=0)
            assert handle.read() == ''
        print(f'Verified seed{seed}:512 episodes,64 exact updates.', flush=True)
    models = {}
    for seed in [11, 22, 33]:
        for mode, filename in [('sampled', 'final.pt'), ('greedy', 'final.pt'), ('initial', 'initial.pt')]:
            path = out/'training'/f'seed{seed}'/filename
            model, _ = from_checkpoint(path); model.load(path)
            models[f'{mode}_{seed}'] = model
    groups = {}
    for filename, expected in summary['trace_hashes'].items():
        assert digest(out/filename) == expected
    with (out/'assessment_traces.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line); check_reward(row)
            groups.setdefault((row['label'], row['scenario']), []).append(row)
    assert len(groups) == 896
    recomputed = {}
    for (label, scenario), rows in groups.items():
        assert [r['day'] for r in rows] == list(range(1, 29))
        expected_start = int(np.random.default_rng(np.random.SeedSequence([scenario, 314159])).integers(15, 1859))
        assert all(r['start'] == expected_start for r in rows)
        if label in models:
            seed = int(label.split('_')[1])
            policy_key = int(np.random.SeedSequence([seed, scenario, 12345]).generate_state(1, dtype=np.uint64)[0])
            generator = torch.Generator().manual_seed(policy_key)
            for row in rows:
                action, lp = models[label].actor.act(row['obs'], row['day'], label.startswith('greedy'), generator)
                np.testing.assert_array_equal(action, row['action']); assert lp == row['log_prob']
        denominator = sum(sum(r['info']['demand']) for r in rows)
        recomputed[label, scenario] = dict(reward_day=sum(r['reward'] for r in rows)/28,
             fill=sum(sum(r['info']['sales']) for r in rows)/denominator if denominator else 1.)
        reference = groups['projected', scenario]
        assert [r['info']['demand'] for r in rows] == [r['info']['demand'] for r in reference]
        assert [r['info']['price'] for r in rows] == [r['info']['price'] for r in reference]
    for record in summary['episode_rows']:
        for metric, value in recomputed[record['label'], record['scenario']].items():
            assert record[metric] == value
    replay = [json.loads(line) for line in (out/'replay_traces.jsonl').read_text().splitlines()]
    assert len(replay) == 168
    for row in replay:
        assert row == groups[row['label'], row['scenario']][row['day']-1]
    # Independent implementation of crossed resampling and endpoint gates.
    names = [(ref, metric) for ref in ['initial', 'projected'] for metric in ['reward_day', 'fill']]
    arrays = []
    for ref, metric in names:
        arrays.append(np.array([[recomputed[f'sampled_{s}', k][metric]-
            recomputed[(f'initial_{s}' if ref == 'initial' else ref), k][metric]
            for k in range(1010000, 1010064)] for s in [11, 22, 33]]))
    rng = np.random.default_rng(20261018); draws = np.empty((10000, 4))
    for b in range(10000):
        rs = rng.integers(0, 3, size=3); cs = rng.integers(0, 64, size=64)
        for col, values in enumerate(arrays):
            draws[b, col] = values[np.ix_(rs, cs)].mean()
    passed = True
    for index, ((ref, metric), values) in enumerate(zip(names, arrays)):
        endpoint = summary['analysis']['endpoints'][f'{ref}_{metric}']
        interval = np.quantile(draws[:, index], [.00625, .99375])
        np.testing.assert_allclose(interval, endpoint['ci9875'], rtol=0, atol=1e-12)
        np.testing.assert_allclose(values.mean(1), endpoint['seed_means'], rtol=0, atol=1e-12)
        assert abs(values.mean()-endpoint['mean']) < 1e-12
        for row, declared in zip(values, endpoint['conditional_seed_ci95']):
            width = t.ppf(.975, 63)*row.std(ddof=1)/8
            np.testing.assert_allclose([row.mean()-width, row.mean()+width], declared, rtol=0, atol=1e-12)
        gate = bool(interval[0] > 0 if metric == 'reward_day' else interval[0] >= -.005)
        assert gate == endpoint['passed']; passed &= gate
        if metric == 'reward_day':
            passed &= bool(np.all(values.mean(1) > 0))
    assert bool(passed) == summary['analysis']['learning_gate']
    assert summary['assessment_and_replay'] == dict(episodes=902, simulated_days=25256)
    assert summary['total_days_including_smoke'] == 68712 and summary['validation_test_episodes'] == 0
    assert summary['assessment_seconds'] <= 600
    write_json(out/'verification.json', dict(status='passed', verifier_sha256=digest(__file__),
               training_updates_reconstructed=192, training_days_replayed_from_archive=43008,
               assessment_episodes=896, simulator_checkpoint_replays=6, new_simulator_days=0,
               paired_demand=True, exact_checkpoint_reconstruction=True, independent_bootstrap_and_gate=True,
               limitation='Archive audit; six simulator replays performed by campaign runner, not additional verifier episodes.'))
    print('Verified paired assessment, six checkpoint replays, independent bootstrap intervals and gate.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True)
    verify(p.parse_args().output)
