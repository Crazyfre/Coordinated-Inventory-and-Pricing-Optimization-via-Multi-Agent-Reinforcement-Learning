"""Recompute saved-label, checkpoint and paired-metric evidence without training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t
from scripts.diagnose_cover_search import CoverPolicy
from src.algos.structured_imitation import InventoryPositionLinearPolicy
from src.algos.structured_residual import StructuredResidualActor


def verify(output):
    out=Path(output);s=json.loads((out/'summary.json').read_text());p=s['protocol']
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert all(digest(path)==p['hashes'][key] for key,path in p['paths'].items())
    assert digest(out/'fitted.pt')==s['checkpoint_sha256']
    assert digest(out/'datasets.npz')==s['dataset_sha256']
    data=np.load(out/'datasets.npz');ck=torch.load(out/'fitted.pt',weights_only=True)
    spec=ck['spec'];n=spec['n'];h=spec['pipe_horizon']
    center=InventoryPositionLinearPolicy.load(p['paths']['center'],n,h)
    target=CoverPolicy(n,h,p['cover'])
    actor=StructuredResidualActor(data['fit_x'].shape[1],n,h,spec['coefficients'])
    # Actor checkpoint storage follows HeterogeneousMAPPO's public format.
    actor.load_state_dict(ck['replenishment'])
    torch.set_num_threads(1)
    traces=[json.loads(line) for line in (out/'traces.jsonl').read_text().splitlines()]
    assert len(traces)==5600==s['trace_days'] and s['episodes']==200
    holdout_accuracy=None
    for label in s['means']:
        x,y=data[label+'_x'],data[label+'_y']
        c=center.predict(x);full=np.asarray([target.act(row)[0] for row in x])
        expected=np.minimum(np.maximum(full,np.maximum(c-2,0)),np.minimum(c+2,40))
        np.testing.assert_array_equal(y,expected)
        rs=[r for r in traces if r['label']==label]
        np.testing.assert_allclose(np.mean(np.asarray([r['actions'] for r in rs])==y),s['agreement'][label])
        if label in ['fit','holdout','greedy']:
            with torch.no_grad():pred=torch.stack([d.probs.argmax(-1) for d in actor(torch.tensor(x))],-1).numpy()
            if label in ['fit','holdout']:
                np.testing.assert_allclose(np.mean(pred==y),s['history'][-1][label],atol=1e-7)
                if label=='holdout':holdout_accuracy=float(np.mean(pred==y))
            else:np.testing.assert_array_equal(pred,np.asarray([r['actions'] for r in rs]))
        episodes=[r for r in s['episode_rows'] if r['label']==label]
        for ep in episodes:
            days=[r for r in rs if r['seed']==ep['seed']];assert len(days)==28
            np.testing.assert_allclose(ep['reward_day'],sum(r['reward'] for r in days)/28)
            np.testing.assert_allclose(ep['fill'],sum(sum(r['sales']) for r in days)/sum(sum(r['demand']) for r in days))
        for key in ['reward_day','fill']:
            np.testing.assert_allclose(s['means'][label][key],np.mean([r[key] for r in episodes]))
    for mode in ['greedy','sampled']:
        actual=[r for r in traces if r['label']==mode]
        assert actual==[dict(r,label=mode) for r in traces if r['label']=='replay_'+mode]
        for ref,comparison in s['comparisons'][mode].items():
            a=[r for r in s['episode_rows'] if r['label']==mode]
            b=[r for r in s['episode_rows'] if r['label']==ref]
            assert [r['seed'] for r in a]==[r['seed'] for r in b]==p['validation_seeds']
            for key,prefix in [('reward_day','reward'),('fill','fill')]:
                delta=np.array([u[key]-v[key] for u,v in zip(a,b)])
                width=t.ppf(.975,19)*delta.std(ddof=1)/np.sqrt(20)
                np.testing.assert_allclose(comparison[prefix+'_delta'],delta.mean())
                np.testing.assert_allclose(comparison[prefix+'_ci95'],[delta.mean()-width,delta.mean()+width])
        c=s['comparisons'][mode]['projected']
        assert s['gates'][mode]==bool(holdout_accuracy>=.99 and s['agreement'][mode]>=.99 and c['reward_ci95'][0]>=-1 and c['fill_ci95'][0]>=-.005)
    assert s['imitation_gate']==all(s['gates'].values())
    seeds=p['fit_seeds']+p['holdout_seeds']
    windows=[int(np.random.default_rng(np.random.SeedSequence([seed,314159])).integers(15,1885-28+2)) for seed in seeds]
    assert s['training_windows']==windows and all(15<=day and day+27<=1885 for day in windows)
    assert set(p['fit_seeds']).isdisjoint(p['holdout_seeds'])
    for label in ['projected','greedy','sampled']:
        assert [r['demand'] for r in traces if r['label']==label]==[r['demand'] for r in traces if r['label']=='center']
    result=dict(passed=True,episodes=200,trace_days=len(traces),checks='hashes, labels, checkpoint greedy predictions, fit/holdout accuracy, all metrics, paired intervals/gates, replay, train windows, paired demands')
    (out/'independent_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');verify(p.parse_args().output)
