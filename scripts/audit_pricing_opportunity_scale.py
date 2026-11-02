"""Descriptive split-future opportunity audit; no policy fit or simulation."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.train_compact_residual import digest,write_json


def describe(gains):
    g=np.asarray(gains,dtype=float)
    if g.ndim!=2 or g.shape[1]!=32 or not len(g) or not np.isfinite(g).all():
        raise ValueError('finite nonempty state by32 paired gains required')
    mean=g.mean(1);a=g[:,:16].mean(1);b=g[:,16:].mean(1)
    # Choosing the sign on one independent future half and scoring the other
    # diagnoses repeatability, but the selector is unavailable to an actor.
    ab=np.where(a>0,b,0);ba=np.where(b>0,a,0)
    return dict(states=len(g),positive_mean=int(sum(mean>0)),both_halves_positive=int(sum((a>0)&(b>0))),
        mean_gain=float(mean.mean()),positive_part_mean=float(np.maximum(mean,0).mean()),
        first_select_count=int(sum(a>0)),first_select_second_score=float(ab.mean()),
        second_select_count=int(sum(b>0)),second_select_first_score=float(ba.mean()),
        symmetric_transfer=float(((ab+ba)/2).mean()),
        median_mc_se=float(np.median(g.std(1,ddof=1)/np.sqrt(32))))


def run(source,output):
    root,out=Path(source),Path(output)
    proof=json.loads((root/'verification.json').read_text());assert proof['status']=='passed'
    freeze=json.loads((root/'assessment_freeze.json').read_text())
    assert digest(root/'policy.json')==freeze['fit_hashes']['policy.json']
    p=json.loads((root/'policy.json').read_text());samples=p['samples']
    assert [(s['episode'],s['rank']) for s in samples]==[(e,r) for e in range(128) for r in [0,1]]
    g=np.array([s['gains'] for s in samples]);base=np.array([p['base_prices'][s['sku']] for s in samples])
    np.testing.assert_allclose(g.mean(1)/base,p['targets'],rtol=0,atol=0)
    groups={}
    for label,mask in [('max_cover',np.arange(256)%2==0),('random_remaining',np.arange(256)%2==1)]:
        groups[label]=describe(g[mask])
    by_sku={str(i):describe(g[[s['sku']==i for s in samples]]) for i in range(10) if any(s['sku']==i for s in samples)}
    by_day={str(d):describe(g[[s['day']==d for s in samples]]) for d in [1,8,15,22]}
    a=g[:,:16].mean(1);b=g[:,16:].mean(1)
    transfer=(np.where(a>0,b,0)+np.where(b>0,a,0))/2
    # Average over the two selected candidates within a window before resampling.
    window=transfer.reshape(128,2).mean(1)
    draws=np.random.default_rng(20261105).integers(128,size=(10000,128))
    ci=np.quantile(window[draws].mean(1),[.025,.975]).tolist()
    rows=[dict(episode=s['episode'],rank=s['rank'],day=s['day'],sku=s['sku'],
        stock=float(s['obs'][s['sku']]),forecast=float(s['obs'][-20+s['sku']]),
        mean=float(g[j].mean()),first=float(a[j]),second=float(b[j]),
        se=float(g[j].std(ddof=1)/np.sqrt(32))) for j,s in enumerate(samples)]
    result=dict(status='complete',units='discounted suffix reward, NOT base-price normalized and NOT reward/day',
        all_selected=describe(g),strata=groups,by_sku=by_sku,by_day=by_day,
        symmetric_transfer_window_bootstrap_ci95=ci,rows=rows,
        new_simulator_days=0,policy_fits=0,
        limitations=['Post-result descriptive archive audit; no confirmatory gate.',
            'Positive-part means use noisy targets and have selection optimism.',
            'Split-future sign selection is not an actor-observable policy.',
            'Two enriched candidates per window do not represent all40 equally; strata reported separately.',
            'Suffix gains overlap in time and cannot be summed into achievable episode gain or a universal policy bound.'],
        hashes={f:digest(root/f) for f in ['policy.json','assessment_freeze.json','verification.json']},
        script_sha256=digest(__file__))
    out.mkdir(parents=True,exist_ok=False);write_json(out/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.source,a.output)
