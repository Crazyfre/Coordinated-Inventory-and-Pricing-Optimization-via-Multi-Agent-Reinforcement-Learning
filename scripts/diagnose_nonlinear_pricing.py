"""Prospectively fixed archive diagnostic, zero environment calls."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from src.algos.diagnostic_regression_tree import fit_tree, predict_tree
from scripts.train_compact_residual import digest, write_json


def metrics(pred, target, halves):
    chosen = np.asarray(pred) > 0
    n = int(chosen.sum())
    windows = len(set((np.flatnonzero(chosen)//2).tolist()))
    means = np.asarray(halves)[chosen].mean(0).tolist() if n else [None, None]
    return dict(mse=float(np.mean((pred-target)**2)), positive_predictions=n,
                positive_windows=windows, positive_targets=int(((target > 0)&chosen).sum()),
                mean_target=float(target[chosen].mean()) if n else None,
                half_means=means,
                continuation_flag=bool(n >= 8 and windows >= 8 and target[chosen].mean() > 0
                                       and min(means) > 0))


def run(source, output):
    root, out = Path(source), Path(output)
    out.mkdir(parents=True, exist_ok=False)
    paths = [root/'policy.json', root/'verification.json', root/'assessment_freeze.json',
             Path(__file__), Path('src/algos/diagnostic_regression_tree.py'),
             Path('NONLINEAR_PRICING_DIAGNOSTIC_PROTOCOL.md')]
    hashes = {str(p): digest(p) for p in paths}
    write_json(out/'manifest.json', dict(hashes=hashes, new_simulator_days=0, fits=128, cap_seconds=120))
    assert json.loads((root/'verification.json').read_text())['status'] == 'passed'
    p = json.loads((root/'policy.json').read_text())
    freeze = json.loads((root/'assessment_freeze.json').read_text())
    assert digest(root/'policy.json') == freeze['fit_hashes']['policy.json']
    assert [(r['episode'],r['rank']) for r in p['samples']] == [(e,j) for e in range(128) for j in [0,1]]
    x = np.asarray(p['matrix'])[:, :16]; y = np.asarray(p['targets'])
    halves = np.array([[np.mean(r['gains'][:16]),np.mean(r['gains'][16:])] for r in p['samples']])
    halves /= np.array([p['base_prices'][r['sku']] for r in p['samples']])[:, None]
    begin = time.perf_counter(); predictions = np.zeros(256); folds=[]
    try:
        for ep in range(128):
            train = np.flatnonzero(np.arange(256)//2 != ep); test = [2*ep,2*ep+1]
            tree = fit_tree(x[train], y[train]); predictions[test] = predict_tree(tree,x[test])
            tree_hash=hashlib.sha256(json.dumps(tree,sort_keys=True).encode()).hexdigest()
            folds.append(dict(episode=ep, train_rows=train.tolist(), test_rows=test, tree=tree,tree_hash=tree_hash))
            if time.perf_counter()-begin > 120: raise TimeoutError('diagnostic cap')
        assert all(digest(path)==h for path,h in hashes.items())
        write_json(out/'folds.json', folds)
        write_json(out/'summary.json', dict(status='complete', predictions=predictions,
            metrics=metrics(predictions,y,halves), fits=128, new_simulator_days=0,
            seconds=time.perf_counter()-begin, folds_hash=digest(out/'folds.json'),
            limitation='Exploratory archive diagnostic, not fresh policy evaluation or PPO/MAPPO.'))
        print(json.dumps(metrics(predictions,y,halves)),flush=True)
    except Exception as exc:
        write_json(out/'failed_status.json',dict(error=str(exc),finished_folds=len(folds))); raise


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();run(args.source,args.output)
