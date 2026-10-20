"""Frozen three-seed compact study: training-only paired assessment, no selection."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.stats import t
from scripts.train_compact_residual import (setup, digest, serial, write_json,
                                            run_training_stage, SOURCES, INPUTS)
from src.env.interface import JointAction
from src.experiments.phase4 import heuristic
from src.algos.imitation_ordering import teacher_indices

SEEDS = [11, 22, 33]
KEYS = list(range(1010000, 1010064))
EXTRA_SOURCES = ['scripts/assess_compact_residual.py', 'scripts/verify_compact_study.py',
                 'tests/test_compact_assessment.py', 'src/experiments/phase4.py']


def action_generator(seed, scenario):
    key = int(np.random.SeedSequence([seed, scenario, 12345]).generate_state(1, dtype=np.uint64)[0])
    return torch.Generator().manual_seed(key)


def paired_interval(values):
    x = np.asarray(values, float)
    width = t.ppf(.975, len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
    return [float(x.mean()-width), float(x.mean()+width)]


def analyze(rows, replicates=10000):
    lookup = {(r['label'], r['scenario']): r for r in rows}
    labels = [f'{mode}_{seed}' for seed in SEEDS for mode in ['sampled', 'greedy', 'initial']]
    labels += ['projected', 'full7.5', 'center', 'constant', 'ss']
    if len(rows) != 896 or len(lookup) != 896 or set(lookup) != {(l, k) for l in labels for k in KEYS}:
        raise ValueError('exactly14 complete paired controllers on64 declared scenarios required')
    matrices = {}
    for reference in ['initial', 'projected']:
        for metric in ['reward_day', 'fill']:
            matrices[f'{reference}_{metric}'] = np.array([
                [lookup[f'sampled_{seed}', k][metric]-lookup[(f'initial_{seed}' if reference == 'initial' else 'projected'), k][metric]
                 for k in KEYS] for seed in SEEDS])
    stack = np.stack(list(matrices.values()))
    rng = np.random.default_rng(20261018)
    bootstrap = np.empty((replicates, 4))
    for i in range(replicates):
        seed_rows = rng.integers(0, 3, 3); columns = rng.integers(0, 64, 64)
        bootstrap[i] = stack[:, seed_rows[:, None], columns[None, :]].mean(axis=(1, 2))
    endpoints = {}
    for i, (name, values) in enumerate(matrices.items()):
        bounds = np.quantile(bootstrap[:, i], [.00625, .99375])
        endpoints[name] = dict(mean=float(values.mean()), ci9875=bounds.tolist(),
                               seed_means=values.mean(1).tolist(),
                               conditional_seed_ci95=[paired_interval(row) for row in values],
                               passed=bool(bounds[0] > 0 if name.endswith('reward_day') else bounds[0] >= -.005))
    all_positive = all(np.all(matrices[f'{r}_reward_day'].mean(1) > 0) for r in ['initial', 'projected'])
    means = {label: {metric: float(np.mean([lookup[label, k][metric] for k in KEYS]))
                     for metric in ['reward_day', 'fill']} for label in labels}
    descriptive = {ref: {metric: float(np.mean([lookup[f'sampled_{seed}', k][metric]-lookup[ref, k][metric]
                                               for seed in SEEDS for k in KEYS]))
                         for metric in ['reward_day', 'fill']}
                   for ref in ['full7.5', 'center', 'constant', 'ss']}
    return dict(endpoints=endpoints, every_seed_reward_positive=bool(all_positive),
                learning_gate=bool(all_positive and all(e['passed'] for e in endpoints.values())),
                means=means, descriptive_sampled_differences=descriptive,
                bootstrap_replicates=replicates, bootstrap_seed=20261018)


def assess_episode(env, model, label, scenario, deadline, counters, trace, seed=None):
    env.reset(seed=scenario)
    generator = action_generator(seed, scenario) if seed is not None else None
    rows = []
    for day in range(1, 29):
        if time.perf_counter() > deadline:
            raise TimeoutError('assessment/replay600s cap')
        if env.pricing_due:
            env.select_price(4)
        obs = env.replenishment_observation().copy()
        if label.startswith(('sampled_', 'initial_', 'greedy_')):
            action, lp = model.actor.act(obs, day, label.startswith('greedy_'), generator)
        elif label == 'projected':
            action = model.actor.support(torch.tensor(obs)[None])[0][0].numpy(); lp = 0.
        elif label == 'full7.5':
            action = teacher_indices(torch.tensor(obs), 10, 7, cover=7.5).numpy(); lp = 0.
        elif label == 'center':
            from src.algos.structured_imitation import InventoryPositionLinearPolicy
            action = InventoryPositionLinearPolicy(10, 7, model.spec['coefficients']).predict(obs); lp = 0.
        else:
            action = heuristic(env, label); lp = 0.
        _, reward, done, info = env.step(JointAction(None, action))
        counters['simulated_days'] += 1
        np.testing.assert_allclose(info['price'], env.base_price*1.2, rtol=0, atol=1e-12)
        assert info['total_on_hand'] <= 500+1e-8
        row = serial(dict(label=label, scenario=scenario, day=day, start=env.sim.start_day_index,
                          obs=obs, action=action, log_prob=lp, reward=reward, info=info))
        trace.write(json.dumps(row, allow_nan=False)+'\n')
        rows.append(row)
        assert done == (day == 28)
    trace.flush(); counters['episodes'] += 1
    demand = sum(sum(r['info']['demand']) for r in rows)
    summary = dict(label=label, scenario=scenario, start=rows[0]['start'],
                   reward_day=sum(r['reward'] for r in rows)/28,
                   fill=sum(sum(r['info']['sales']) for r in rows)/demand if demand else 1.,
                   components_day={term: sum(r['info']['reward_terms'][term] for r in rows)/28
                                   for term in rows[0]['info']['reward_terms']})
    return summary, rows


def run(output, smoke):
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    hashes = {p: digest(p) for p in SOURCES+list(INPUTS.values())+EXTRA_SOURCES}
    write_json(out/'manifest.json', dict(hashes=hashes, seeds=SEEDS, assessment_keys=KEYS,
               primary='sampled', mode='training_only', smoke=str(smoke), bootstrap_seed=20261018,
               bootstrap_replicates=10000, interval_quantiles=[.00625, .99375],
               policy_sampling='uint64 SeedSequence([training_seed,scenario,12345])',
               validation_test_episodes=0))
    counters = dict(episodes=0, simulated_days=0)
    stage = 'training'; started = time.perf_counter()
    try:
        run_training_stage(out/'training', smoke)
        stage = 'assessment'
        assessed_start = time.perf_counter(); deadline = assessed_start+600
        env, baseline = setup(11)
        models = {}; checkpoint_hashes = {}
        for seed in SEEDS:
            for name, filename in [('sampled', 'final.pt'), ('initial', 'initial.pt')]:
                _, model = setup(seed)
                path = out/'training'/f'seed{seed}'/filename
                model.load(path); models[f'{name}_{seed}'] = model
                checkpoint_hashes[str(path)] = digest(path)
            models[f'greedy_{seed}'] = models[f'sampled_{seed}']
        write_json(out/'frozen_checkpoints.json', checkpoint_hashes)
        labels = ['projected', 'full7.5', 'center', 'constant', 'ss']
        labels += [f'{mode}_{seed}' for seed in SEEDS for mode in ['sampled', 'greedy', 'initial']]
        summaries, replay_targets = [], {}
        with (out/'assessment_traces.jsonl').open('w') as trace:
            for key in KEYS:
                demand_reference = None
                for label in labels:
                    seed = int(label.split('_')[1]) if '_' in label else None
                    result, rows = assess_episode(env, models.get(label, baseline), label, key,
                                                 deadline, counters, trace, seed)
                    demand = [r['info']['demand'] for r in rows]
                    if demand_reference is None:
                        demand_reference = demand
                    assert demand == demand_reference
                    summaries.append(result)
                    if key == KEYS[0] and label.startswith(('sampled_', 'greedy_')):
                        replay_targets[label] = rows
                write_json(out/'assessment_progress.json', dict(completed_scenarios=key-KEYS[0]+1, **counters))
        with (out/'replay_traces.jsonl').open('w') as trace:
            for label, expected in replay_targets.items():
                seed = int(label.split('_')[1]); _, restored = setup(seed)
                restored.load(out/'training'/f'seed{seed}'/'final.pt')
                _, replay = assess_episode(env, restored, label, KEYS[0], deadline, counters, trace, seed)
                assert replay == expected, label
        for path, expected in {**hashes, **checkpoint_hashes}.items():
            assert digest(path) == expected, path
        analysis = analyze(summaries)
        if time.perf_counter() > deadline:
            raise TimeoutError('assessment/replay cap including bootstrap')
        write_json(out/'summary.json', dict(status='complete', analysis=analysis, episode_rows=summaries,
                   assessment_and_replay=counters, training_episodes=1536, training_days=43008,
                   smoke_episodes=16, total_episodes_including_smoke=2454, total_days_including_smoke=68712,
                   seconds=time.perf_counter()-started, assessment_seconds=time.perf_counter()-assessed_start,
                   demand_pairing=True, six_checkpoint_replays=True, validation_test_episodes=0,
                   trace_hashes={p: digest(out/p) for p in ['assessment_traces.jsonl', 'replay_traces.jsonl']},
                   training_trace_hashes={str(seed): digest(out/'training'/f'seed{seed}'/'traces.jsonl') for seed in SEEDS}))
        print(json.dumps(dict(status='complete', learning_gate=analysis['learning_gate'],
                              endpoints=analysis['endpoints'], seconds=time.perf_counter()-started)), flush=True)
    except Exception as exc:
        write_json(out/'failed_status.json', dict(stage=stage, status='failed_partial', error=str(exc),
                   assessment_and_replay=counters, seconds=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True); p.add_argument('--verified-smoke', required=True)
    args = p.parse_args(); run(args.output, args.verified_smoke)
