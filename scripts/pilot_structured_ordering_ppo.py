"""Fixed-budget daily cover PPO with original fixed1.20 pricing."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.algos.structured_ordering_ppo import StructuredOrderingPPO, ordering_features, execute_cover
from src.algos.structured_pricing_ppo import critic_features
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.env.interface import JointAction
from src.eval.parameter_review import episode_metrics
from src.experiments.research import make_research_env
from scripts.train_clean_research import TrainingWindows
from scripts.train_compact_residual import INPUTS, digest, serial, write_json

STUDY = Path('outputs/rl_discovery/parameter_confirmation_v1')


def collect(env, model, covers, key, label, handle, deadline, greedy=False, baseline=False):
    env.reset(seed=key)
    trace, decisions = [], []
    for day in range(1, 29):
        if time.perf_counter() > deadline:
            raise TimeoutError('ordering PPO time cap reached')
        if env.pricing_due:
            env.select_price(np.full(10, 4))
        obs = env.replenishment_observation().copy()
        raw_state = env.get_state().copy()
        f, state = ordering_features(obs, day), critic_features(raw_state, day)
        latent, lp = (covers.astype(np.float32), None) if baseline else model.actor.act(f, greedy)
        action = execute_cover(obs, latent)
        decision = serial(dict(features=f, state=state, raw_state=raw_state,
            action=latent, logp=lp, value=model.value(state)))
        decisions.append(decision)
        _, reward, done, info = env.step(JointAction(None, action))
        assert done == (day == 28)
        row = serial(dict(label=label, scenario=key, day=day, start=int(env.sim.start_day_index),
            decision=decision, obs=obs, actions=action, reward=reward, info=info,
            demand_seed=int(env.sim.scenario_seed), supplier_seed=int(env.pipeline.scenario_seed)))
        handle.write(json.dumps(row, allow_nan=False) + '\n'); trace.append(row)
    handle.flush()
    return dict(decisions=decisions, rewards=[r['reward'] for r in trace]), episode_metrics(trace), trace


def analyze(rows):
    index = {(r['label'], r['scenario']): r for r in rows}
    assert len(index) == len(rows) == 256
    draws = np.random.default_rng(20261026).integers(64, size=(10000, 64))
    result = {}
    for label in ['greedy', 'sampled']:
        for ref in ['baseline', 'initial']:
            for metric in ['reward_day', 'fill']:
                delta = np.array([index[label, k][metric] - index[ref, k][metric]
                                  for k in range(2260000, 2260064)])
                result[f'{label}-{ref}:{metric}'] = dict(mean=float(delta.mean()),
                    ci95=np.quantile(delta[draws].mean(1), [.025, .975]).tolist())
    return result


def run(output):
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    assert json.loads((STUDY / 'verification.json').read_text())['confirmed_gate']
    covers = np.array(json.loads((STUDY / 'sku_seed211_final.json').read_text())['mean'])
    sources = sorted({str(p) for folder in ['src', 'scripts'] for p in Path(folder).rglob('*.py')}
        | {INPUTS['calibration'], str(STUDY / 'sku_seed211_final.json'), 'STRUCTURED_ORDERING_PPO_PROTOCOL.md'})
    hashes = {p: digest(p) for p in sources}
    torch.set_num_threads(1); start = time.perf_counter(); count = 0
    try:
        calibration = load_identifiable_training_calibration(INPUTS['calibration'])
        env = TrainingWindows(make_research_env('S2', split='train', start=15, days=28,
            params=calibration, pricing_scope='per_sku'), 28)
        env.reset(seed=0)
        model = StructuredOrderingPPO(len(env.get_state()), covers, 701)
        initial = StructuredOrderingPPO(len(env.get_state()), covers, 701)
        torch.save(model.state(), out / 'initial.pt')
        write_json(out / 'manifest.json', dict(hashes=hashes, covers=covers, state_dim=len(env.get_state()),
            base_prices=env.base_price, seed=701, updates=64, episodes_per_update=16,
            maximum_seconds=1800, planned_episodes=1282, test_scoring=False))
        history = []
        with (out / 'training.jsonl').open('w') as handle:
            for u in range(64):
                batch = []
                for e in range(16):
                    data, _, _ = collect(env, model, covers, 2250000 + u * 16 + e,
                        f'u{u}', handle, start + 1800)
                    batch.append(data); count += 1
                torch.save(dict(batch=batch, rng=torch.get_rng_state()), out / f'batch_{u:03d}.pt')
                metrics = model.update(batch)
                torch.save(model.state(), out / f'update_{u + 1:03d}.pt')
                history.append(metrics)
                write_json(out / 'progress.json', dict(updates=u + 1, episodes=count, metrics=metrics))
                if (u + 1) % 8 == 0:
                    print(f'ordering PPO: {u + 1}/64 updates', flush=True)
        frozen = {p: digest(out / p) for p in ['initial.pt', 'update_064.pt']}
        write_json(out / 'assessment_freeze.json', frozen)
        rows, targets = [], {}
        with (out / 'assessment.jsonl').open('w') as handle:
            for key in range(2260000, 2260064):
                for label in ['baseline', 'initial', 'greedy', 'sampled']:
                    torch.manual_seed(key + 100000)
                    actor = initial if label == 'initial' else model
                    _, result, trace = collect(env, actor, covers, key, label, handle, start + 1800,
                        greedy=label == 'greedy', baseline=label == 'baseline')
                    count += 1; rows.append(result)
                    if key == 2260000 and label in ['greedy', 'sampled']:
                        targets[label] = trace
        with (out / 'replays.jsonl').open('w') as handle:
            for label in ['greedy', 'sampled']:
                torch.manual_seed(2360000)
                _, _, trace = collect(env, model, covers, 2260000, label, handle, start + 1800,
                    greedy=label == 'greedy')
                count += 1; assert trace == targets[label]
        assert count == 1282 and all(digest(p) == h for p, h in hashes.items())
        assert all(digest(out / p) == h for p, h in frozen.items())
        result = analyze(rows)
        write_json(out / 'summary.json', dict(status='complete', rows=rows, effects=result,
            history=history, episodes=count, days=count * 28, seconds=time.perf_counter() - start,
            test_scoring=False, trace_hashes={p.name: digest(p) for p in out.glob('*.jsonl')},
            archive_hashes={p.name: digest(p) for p in out.glob('*.pt')}))
        print(json.dumps(result), flush=True)
    except Exception as exc:
        write_json(out / 'failed_status.json', dict(error=str(exc), complete_episodes=count))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    run(p.parse_args().output)
