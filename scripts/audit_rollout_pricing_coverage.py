"""Descriptive coverage and noise audit; no policy refit or simulation."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.train_compact_residual import digest,write_json


def run(source,output):
    root=Path(source);out=Path(output)
    proof=json.loads((root/'verification.json').read_text());summary=json.loads((root/'summary.json').read_text())
    assert proof['status']=='passed'
    for p,h in summary['trace_hashes'].items():assert digest(root/p)==h
    records=[];regions={str(t):[] for t in [.5,1.,1.5,2.,2.5,3.]}
    with (root/'fit_states.jsonl').open() as handle:
        for line in handle:
            r=json.loads(line);x=np.array(r['statistics']['obs'],np.float32)
            values=np.array([[b['value'] for b in d['branches']] for d in r['draws']])
            gains=values[:,1:]-values[:,0,None]
            np.testing.assert_array_equal(gains,r['statistics']['advantages'])
            for sku in range(10):
                g=gains[:,sku];mean=float(g.mean());se=float(g.std(ddof=1)/4)
                first=float(g[:8].mean());second=float(g[8:].mean())
                row=dict(episode=r['episode'],day=r['day'],sku=sku,stock=float(x[sku]),forecast=float(x[-20+sku]),
                    mean_gain=mean,mc_se=se,first_half=first,second_half=second,
                    both_halves_positive=bool(first>0 and second>0))
                records.append(row)
                for t in [.5,1.,1.5,2.,2.5,3.]:
                    if x[sku]>np.float32(t*7)*x[-20+sku]:regions[str(t)].append(row)
    assert len(records)==640
    def describe(rows):
        return dict(state_sku_pairs=len(rows),independent_windows=len({r['episode'] for r in rows}),
            positive_means=sum(r['mean_gain']>0 for r in rows),
            both_halves_positive=sum(r['both_halves_positive'] for r in rows),
            mean_gain=float(np.mean([r['mean_gain'] for r in rows])) if rows else None,
            median_mc_se=float(np.median([r['mc_se'] for r in rows])) if rows else None)
    counts={};episodes={}
    with (root/'assessment.jsonl').open() as handle:
        for line in handle:
            r=json.loads(line)
            if r['label']!='baseline':continue
            for row in r['trace']:
                if row['pricing_obs'] is None:continue
                x=np.array(row['pricing_obs'],np.float32)
                for sku in range(10):
                    for t in [1.5,2.,2.5,3.]:
                        key=f'{sku}:{t}';counts.setdefault(key,0);episodes.setdefault(key,set())
                        if x[sku]>np.float32(t*7)*x[-20+sku]:
                            counts[key]+=1;episodes[key].add(r['key'])
    result=dict(status='complete',all_states=describe(records),regions={t:describe(v) for t,v in regions.items()},
        high_cover_states=regions['1.5'],assessment_baseline_coverage={k:dict(decisions=v,windows=len(episodes[k])) for k,v in counts.items()},
        new_simulator_days=0,new_policy_fits=0,
        limitation='Post-result descriptive audit. Six region summaries are not confirmatory tests. Conditional sample means and half agreement are not true action values; states/SKUs in one window are dependent. Assessment coverage is descriptive and must not tune this frozen policy.',
        hashes={p:digest(root/p) for p in ['summary.json','verification.json','fit_states.jsonl','assessment.jsonl']},script_sha256=digest(__file__))
    out.mkdir(parents=True,exist_ok=False);write_json(out/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['assessment_baseline_coverage','hashes']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.source,a.output)
