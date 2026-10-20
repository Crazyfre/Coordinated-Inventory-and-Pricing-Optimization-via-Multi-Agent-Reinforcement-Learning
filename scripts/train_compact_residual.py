"""Bounded compact PPO training stage. Default CLI permits only the timing smoke."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

from src.algos.compact_residual import CompactResidualPPO
from src.algos.structured_imitation import InventoryPositionLinearPolicy
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env
from src.env.interface import JointAction
from scripts.train_clean_research import TrainingWindows

INPUTS = dict(center='outputs/structured_imitation/seed11_v1/structured_policy.json',
              calibration='outputs/calibration_correction/identifiable_v2/train_only_nb_identifiable_v2.json',
              selection='outputs/cover_search/bounded_v1/selection.json')
SOURCES = ['src/algos/compact_residual.py', 'scripts/train_compact_residual.py',
           'src/algos/ppo.py', 'src/algos/structured_imitation.py', 'src/algos/imitation_ordering.py',
           'src/algos/stability.py', 'src/env/joint_env.py', 'src/env/replenishment_env.py',
           'src/experiments/research.py', 'scripts/train_clean_research.py',
           'COMPACT_RESIDUAL_SPEC.md', 'COMPACT_RESIDUAL_PROTOCOL.md']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(serial(value), indent=2, allow_nan=False))


def setup(seed):
    params = load_identifiable_training_calibration(INPUTS['calibration'])
    if json.loads(Path(INPUTS['selection']).read_text())['selected']['cover'] != 7.5:
        raise ValueError('frozen selected cover must be7.5')
    env = TrainingWindows(make_research_env('S2', split='train', start=15, days=28,
                                           params=params, pricing_scope='shared'), 28)
    env.reset(seed=0)  # constructor-only reset, no simulation
    center = InventoryPositionLinearPolicy.load(INPUTS['center'], 10, 7)
    provenance = {k: digest(p) for k, p in INPUTS.items()}
    model = CompactResidualPPO(len(env.get_state()), center.coefficients, env.sim.series_ids, provenance, seed)
    return env, model


def collect_episode(env, model, scenario, trace, deadline, counters, *, greedy=False):
    env.reset(seed=scenario)
    rows = []
    while True:
        if time.perf_counter() > deadline:
            raise TimeoutError('declared stage deadline exceeded')
        if env.pricing_due:
            env.select_price(4)
        day = env.day
        obs = env.replenishment_observation().copy()
        state = np.r_[env.get_state(), (29-day)/28].astype(np.float32)
        default = model.actor.support(torch.as_tensor(obs)[None])[0][0].numpy()
        action, lp = model.actor.act(obs, day, greedy, model.generator)
        value = model.value(state)
        _, reward, done, info = env.step(JointAction(None, action))
        counters['simulated_days'] += 1
        np.testing.assert_allclose(info['price'], env.base_price*1.2, rtol=0, atol=1e-12)
        if info['total_on_hand'] > 500+1e-8:
            raise AssertionError('on-hand capacity violated')
        row = dict(scenario=int(scenario), start=int(env.sim.start_day_index), day=int(day),
                   update=model.updates, obs=obs, state=state, action=action, default=default,
                   log_prob=lp, value=value, reward=float(reward), info=info)
        trace.write(json.dumps(serial(row), allow_nan=False)+'\n'); trace.flush()
        rows.append(row)
        if done:
            break
    if len(rows) != 28:
        raise AssertionError('expected complete28-day episode')
    counters['episodes'] += 1
    return rows


def batch_diagnostics(episodes):
    rows = [r for ep in episodes for r in ep]
    changed = np.stack([r['action'] != r['default'] for r in rows])
    phases = [(1, 7), (8, 14), (15, 21), (22, 26), (27, 28)]
    return dict(nondefault_by_sku=changed.sum(0).tolist(),
                nondefault_by_phase=[int(changed[[lo <= r['day'] <= hi for r in rows]].sum()) for lo, hi in phases],
                reward_day=float(np.mean([r['reward'] for r in rows])),
                episode_rows=[dict(scenario=ep[0]['scenario'], start=ep[0]['start'],
                                   reward_day=sum(r['reward'] for r in ep)/28,
                                   fill=sum(r['info']['sales'].sum() for r in ep)/max(sum(r['info']['demand'].sum() for r in ep), 1e-12))
                              for ep in episodes])


def run_smoke(output):
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    hashes = {p: digest(p) for p in SOURCES+list(INPUTS.values())}
    write_json(out/'manifest.json', dict(variant=CompactResidualPPO.variant, hashes=hashes,
               seed=44, scenario_keys=list(range(1003000, 1003016)), stage='timing_smoke',
               updates=2, episodes=16, max_seconds=60, validation_test_episodes=0))
    counters = dict(episodes=0, simulated_days=0)
    started = time.perf_counter()
    try:
        env, model = setup(44)
        setup_seconds = time.perf_counter()-started
        model.save(out/'initial.pt')
        history = []
        with (out/'traces.jsonl').open('w') as trace:
            for update in range(2):
                episodes = [collect_episode(env, model, key, trace, started+60, counters)
                            for key in range(1003000+8*update, 1003008+8*update)]
                if time.perf_counter() > started+60:
                    raise TimeoutError('smoke deadline before update')
                history.append(dict(**model.update(episodes), **batch_diagnostics(episodes)))
                model.save(out/f'update_{update+1}.pt')
                write_json(out/'progress.json', dict(counters=counters, history=history))
        elapsed = time.perf_counter()-started
        if elapsed > 60:
            raise TimeoutError('smoke exceeded60s including setup, updates and checkpoints')
        if {p: digest(p) for p in hashes} != hashes:
            raise AssertionError('source or inputs changed during smoke')
        # Includes setup and checkpoints, deliberately loaded throughput estimate.
        projected = elapsed/16*1536
        write_json(out/'summary.json', dict(status='complete', **counters, seconds=elapsed,
                   setup_seconds=setup_seconds, projected_training_seconds=projected,
                   training_cap_seconds=1800, budget_fits=projected <= 1800,
                   actor_parameters=sum(p.numel() for p in model.actor.parameters()),
                   critic_parameters=sum(p.numel() for p in model.critic.parameters()), history=history,
                   validation_test_episodes=0, campaign_executed=False, weights_disposable=True,
                   checkpoint_hashes={p.name: digest(p) for p in out.glob('*.pt')}, trace_sha256=digest(out/'traces.jsonl')))
        print(json.dumps(dict(status='complete', seconds=elapsed, projected_training_seconds=projected,
                              budget_fits=projected <= 1800, **counters)), flush=True)
    except Exception as exc:
        write_json(out/'failed_status.json', dict(status='failed_partial', error=str(exc), **counters,
                   seconds=time.perf_counter()-started, campaign_executed=False))
        raise


def run_training_stage(output, smoke_output):
    """Explicit future campaign entry point; never called by the timing smoke."""
    smoke = Path(smoke_output)
    summary = json.loads((smoke/'summary.json').read_text())
    verification = json.loads((smoke/'verification.json').read_text())
    manifest = json.loads((smoke/'manifest.json').read_text())
    if summary['status'] != 'complete' or not summary['budget_fits'] or verification['status'] != 'passed':
        raise ValueError('completed verified smoke within budget required')
    for path, expected in manifest['hashes'].items():
        if digest(path) != expected:
            raise ValueError('source/input changed since verified smoke: '+path)
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.perf_counter(); counters = dict(episodes=0, simulated_days=0)
    write_json(out/'manifest.json', dict(stage='training_only', hashes=manifest['hashes'],
               smoke_summary_sha256=digest(smoke/'summary.json'), seeds=[11, 22, 33],
               episodes_per_seed=512, max_seconds=1800, assessment_executed=False))
    completed = []
    try:
        for seed, first_key in [(11, 1000000), (22, 1001000), (33, 1002000)]:
            env, model = setup(seed)
            seed_out = out/f'seed{seed}'; seed_out.mkdir()
            model.save(seed_out/'initial.pt'); history = []
            with (seed_out/'traces.jsonl').open('w') as trace:
                for update in range(64):
                    episodes = [collect_episode(env, model, key, trace, started+1800, counters)
                                for key in range(first_key+8*update, first_key+8*(update+1))]
                    if time.perf_counter() > started+1800:
                        raise TimeoutError('training deadline before update')
                    history.append(dict(**model.update(episodes), **batch_diagnostics(episodes)))
                    if update+1 in (16, 32, 48, 64):
                        model.save(seed_out/('final.pt' if update == 63 else f'episode{8*(update+1)}.pt'))
                    write_json(seed_out/'history.json', history)
                    write_json(out/'progress.json', dict(seed=seed, updates=update+1, **counters))
            completed.append(seed)
        if time.perf_counter() > started+1800:
            raise TimeoutError('training cap exceeded')
        for path, expected in manifest['hashes'].items():
            if digest(path) != expected:
                raise AssertionError('source/input changed during training')
        write_json(out/'summary.json', dict(status='training_complete', completed_seeds=completed,
                   **counters, seconds=time.perf_counter()-started, assessment_executed=False,
                   validation_test_episodes=0))
    except Exception as exc:
        write_json(out/'failed_status.json', dict(status='failed_partial', error=str(exc),
                   completed_seeds=completed, **counters, seconds=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=['smoke', 'training'], default='smoke')
    parser.add_argument('--verified-smoke')
    args = parser.parse_args()
    if args.stage == 'training':
        if not args.verified_smoke:
            parser.error('--verified-smoke is required for training')
        run_training_stage(args.output, args.verified_smoke)
    else:
        run_smoke(args.output)
