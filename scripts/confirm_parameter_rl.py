"""Conditional five-seed frozen parameter-space RL confirmation cohort."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.stats import t
from scripts.discover_parameter_rl import train_arm, episode
from scripts.train_compact_residual import setup, digest, write_json, INPUTS
from src.algos.parameter_policy_gradient import initial_mean, sigmas
from src.experiments.research import make_research_env
from src.demand.identifiable_calibration import load_identifiable_training_calibration

SEEDS = [211, 222, 233, 244, 255]
KEYS = list(range(1300000, 1300128))


def analyze(rows):
    index = {(r['label'], r['scenario']): r for r in rows}
    assert len(index) == len(rows) == 1792
    matrices = {}
    for ref in ['full7_5', 'initial']:
        for metric in ['reward_day', 'fill']:
            matrices[f'{ref}_{metric}'] = np.array([[index[f'learned_{s}', k][metric]-
                index[(f'initial_{s}' if ref == 'initial' else ref), k][metric] for k in KEYS] for s in SEEDS])
    values = np.stack(list(matrices.values())); rng = np.random.default_rng(20261019)
    draws = np.empty((10000, 4))
    for b in range(10000):
        r = rng.integers(0, 5, 5); c = rng.integers(0, 128, 128)
        draws[b] = values[:, r[:, None], c[None, :]].mean((1, 2))
    endpoints = {}
    for j, (name, array) in enumerate(matrices.items()):
        ci = np.quantile(draws[:, j], [.00625, .99375])
        intervals = []
        for row in array:
            width = t.ppf(.975, 127)*row.std(ddof=1)/np.sqrt(128)
            intervals.append([float(row.mean()-width), float(row.mean()+width)])
        endpoints[name] = dict(mean=float(array.mean()), ci9875=ci.tolist(), seed_means=array.mean(1).tolist(),
             conditional_seed_ci95=intervals, passed=bool(ci[0] > 0 if name.endswith('reward_day') else ci[0] >= -.005))
    positive = all(np.all(matrices[f'{ref}_reward_day'].mean(1) > 0) for ref in ['full7_5', 'initial'])
    labels = sorted(set(r['label'] for r in rows))
    means = {label: {metric: float(np.mean([index[label, k][metric] for k in KEYS])) for metric in ['reward_day', 'fill']}
             for label in labels}
    return dict(endpoints=endpoints, every_seed_reward_positive=bool(positive),
                learning_gate=bool(positive and all(v['passed'] for v in endpoints.values())), means=means)


def run(output, pilot):
    pilot = Path(pilot); chosen = json.loads((pilot/'summary.json').read_text())['selected']
    if chosen not in ['sku', 'arrival'] or json.loads((pilot/'verification.json').read_text())['status'] != 'passed':
        raise ValueError('verified eligible development arm required')
    out = Path(output); out.mkdir(parents=True, exist_ok=False); torch.set_num_threads(1)
    paths = ['PARAMETER_RL_CONFIRMATION_PROTOCOL.md', 'RL_DISCOVERY_PROTOCOL.md',
             'src/algos/parameter_policy_gradient.py', 'scripts/discover_parameter_rl.py',
             'scripts/confirm_parameter_rl.py', 'scripts/verify_parameter_confirmation.py',
             'src/experiments/research.py', 'src/env/joint_env.py', 'src/env/replenishment_env.py',
             'scripts/train_compact_residual.py', 'scripts/train_clean_research.py']+list(INPUTS.values())
    hashes = {p: digest(p) for p in paths}
    write_json(out/'manifest.json', dict(hashes=hashes, arm=chosen, seeds=SEEDS, keys=KEYS,
               pilot_summary_sha256=digest(pilot/'summary.json'), pilot_verification_sha256=digest(pilot/'verification.json'),
               updates=64, pairs=8, training_max_seconds=3600, assessment_max_seconds=1500,
               primary_mode='deterministic_mean', split='validation', test_scoring=False))
    started = time.perf_counter(); counters = dict(episodes=0, days=0); stage='training'
    try:
        env, reference = setup(SEEDS[0]); learners = {}
        with (out/'training_traces.jsonl').open('w') as trace:
            for j, seed in enumerate(SEEDS):
                learners[seed] = train_arm(env, chosen, seed, 1200000+j*10000, 64, 8,
                                          out, trace, counters, started+3600)
        train_seconds = time.perf_counter()-started
        assert train_seconds <= 3600
        frozen = {str(seed): digest(out/f'{chosen}_seed{seed}_final.json') for seed in SEEDS}
        write_json(out/'frozen_checkpoints.json', frozen)
        write_json(out/'training_complete.json', dict(seconds=train_seconds, counters=counters.copy()))
        stage='assessment'; assessed_start = time.perf_counter(); deadline = assessed_start+1500
        params = load_identifiable_training_calibration(INPUTS['calibration'])
        env = make_research_env('S2', split='validation', start=1886, days=28, params=params, pricing_scope='shared')
        rows, replay_targets = [], {}
        with (out/'assessment_traces.jsonl').open('w') as trace:
            for key in KEYS:
                policies = {f'learned_{s}': learners[s].mean for s in SEEDS}
                for seed in SEEDS:
                    rng = np.random.default_rng(np.random.SeedSequence([seed, key, 707]))
                    policies[f'initial_{seed}'] = initial_mean(chosen)+sigmas(chosen)*rng.normal(size=len(initial_mean(chosen)))
                policies.update(full7_5=initial_mean('sku'), projected='projected', constant='constant', ss='ss')
                demand = None
                for label, policy in policies.items():
                    result, traces = episode(env, policy, key, label, trace, counters, deadline, reference)
                    current = [r['info']['demand'] for r in traces]
                    if demand is None:
                        demand = current
                    assert demand == current
                    rows.append(result)
                    if key == KEYS[0] and label.startswith(('learned_', 'initial_')):
                        replay_targets[label] = traces
                write_json(out/'assessment_progress.json', dict(scenarios=key-KEYS[0]+1, counters=counters))
        with (out/'replay_traces.jsonl').open('w') as trace:
            for label, expected in replay_targets.items():
                seed = int(label.split('_')[1])
                if label.startswith('learned_'):
                    policy = np.array(json.loads((out/f'{chosen}_seed{seed}_final.json').read_text())['mean'])
                else:
                    rng = np.random.default_rng(np.random.SeedSequence([seed, KEYS[0], 707]))
                    policy = initial_mean(chosen)+sigmas(chosen)*rng.normal(size=len(initial_mean(chosen)))
                _, replay = episode(env, policy, KEYS[0], label, trace, counters, deadline, reference)
                assert replay == expected
        result = analyze(rows)
        assert time.perf_counter() <= deadline
        for p, h in hashes.items():
            assert digest(p) == h
        for seed in SEEDS:
            assert digest(out/f'{chosen}_seed{seed}_final.json') == frozen[str(seed)]
        write_json(out/'summary.json', dict(status='complete', arm=chosen, analysis=result, rows=rows,
                   counters=counters, training_seconds=train_seconds, assessment_seconds=time.perf_counter()-assessed_start,
                   seconds=time.perf_counter()-started, exact_replays=10, test_scoring=False,
                   trace_hashes={n: digest(out/n) for n in ['training_traces.jsonl', 'assessment_traces.jsonl', 'replay_traces.jsonl']}))
        print(json.dumps(dict(arm=chosen, gate=result['learning_gate'], endpoints=result['endpoints'], counters=counters)), flush=True)
    except Exception as exc:
        write_json(out/'failed_status.json', dict(stage=stage, error=str(exc), counters=counters, seconds=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); p.add_argument('--pilot', required=True)
    args = p.parse_args(); run(args.output, args.pilot)
