"""Read-only descriptive critic/actor audit of completed PPO pilot archives."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scripts.train_compact_residual import digest, write_json


def audit(root, kind):
    root = Path(root)
    verification = json.loads((root / 'verification.json').read_text())
    assert verification['status'] == 'passed'
    summary = json.loads((root / 'summary.json').read_text())
    for p, h in summary['archive_hashes'].items(): assert digest(root / p) == h
    initial = torch.load(root / 'initial.pt', weights_only=True, map_location='cpu')['actor']
    final = torch.load(root / 'update_064.pt', weights_only=True, map_location='cpu')['actor']
    records = []
    for update in range(64):
        batch = torch.load(root / f'batch_{update:03d}.pt', weights_only=True, map_location='cpu')['batch']
        values, mc, features = [], [], []
        for episode in batch:
            r = np.array(episode['rewards']) / 100
            for j, d in enumerate(episode['decisions']):
                day = j if kind == 'ordering' else j * 7
                mc.append(sum(.99 ** k * reward for k, reward in enumerate(r[day:])))
                values.append(d['value']); features.append(d['features'])
        v, target = np.array(values), np.array(mc)
        variance = float(np.var(target))
        records.append(dict(update=update + 1, mc_target_variance=variance,
            mc_explained_variance=float(1 - np.var(target - v) / variance) if variance else None,
            mean_value_error=float(np.mean(v - target)), rmse=float(np.sqrt(np.mean((v - target) ** 2)))))
    features = torch.tensor(features, dtype=torch.float32)
    if kind == 'ordering':
        old = initial['covers'] + (features * initial['weight']).sum(-1)
        new = final['covers'] + (features * final['weight']).sum(-1)
        actor = dict(mean_absolute_cover_shift=float((new - old).abs().mean()),
            max_absolute_cover_shift=float((new - old).abs().max()),
            final_latent_mean_outside_cover_bounds=float(((new < 0) | (new > 20)).float().mean()))
    else:
        old = torch.einsum('bif,ifk->bik', features, initial['weight']).softmax(-1)
        new = torch.einsum('bif,ifk->bik', features, final['weight']).softmax(-1)
        actor = dict(initial_index4_probability=float(old[..., 4].mean()),
            final_index4_probability=float(new[..., 4].mean()),
            final_greedy_non4_fraction=float((new.argmax(-1) != 4).float().mean()))
    return dict(kind=kind, root=str(root), verification_hash=digest(root / 'verification.json'),
        summary_hash=digest(root / 'summary.json'), per_update=records, actor_on_final_batch=actor,
        first8_mean_mc_explained_variance=float(np.mean([r['mc_explained_variance'] for r in records[:8]])),
        last8_mean_mc_explained_variance=float(np.mean([r['mc_explained_variance'] for r in records[-8:]])),
        limitation='Descriptive noisy Monte Carlo target fit on on-policy training batches; not causal diagnosis or test evidence.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False); torch.set_num_threads(1)
    result = {kind: audit(f'outputs/structured_mappo/{kind}_ppo_pilot_v1', kind)
              for kind in ['pricing', 'ordering']}
    write_json(out / 'audit.json', dict(results=result, new_simulator_days=0, source_sha256=digest(__file__)))
    print(json.dumps({k: {f: v for f, v in r.items() if f != 'per_update'} for k, r in result.items()}), flush=True)
