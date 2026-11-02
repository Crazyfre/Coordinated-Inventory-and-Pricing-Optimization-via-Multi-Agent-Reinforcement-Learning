"""Post-result subgroup economics; no simulation, fitting or policy claims."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.audit_pricing_opportunity_scale import describe
from scripts.train_compact_residual import digest,write_json

TERMS=['revenue','ordering_cost','holding_cost','stockout_cost','price_change_cost']


def branch_economics(branches):
    assert [b['action'] for b in branches]==[3,4]
    a,b=[v['trace'] for v in branches]
    assert len(a)==len(b)
    delta={k:sum(.99**i*(x['info']['reward_terms'][k]-y['info']['reward_terms'][k])
                 for i,(x,y) in enumerate(zip(a,b))) for k in TERMS}
    gain=branches[0]['value']-branches[1]['value']
    assert abs(delta['revenue']-sum(delta[k] for k in TERMS[1:])-gain)<1e-8
    return delta


def run(source,output):
    root,out=Path(source),Path(output)
    proof=json.loads((root/'verification.json').read_text());assert proof['status']=='passed'
    freeze=json.loads((root/'assessment_freeze.json').read_text())
    for name in ['policy.json','states.jsonl']:assert digest(root/name)==freeze['fit_hashes'][name]
    policy=json.loads((root/'policy.json').read_text())
    expected={(r['episode'],r['rank']):r for r in policy['samples']};rows=[];seen=set()
    with (root/'states.jsonl').open() as f:
        for line in f:
            record=json.loads(line);s=record['sample'];key=(s['episode'],s['rank'])
            assert key not in seen and s==expected[key];seen.add(key)
            economic=[branch_economics(d['branches']) for d in record['draws']]
            np.testing.assert_array_equal(s['gains'],[d['branches'][0]['value']-d['branches'][1]['value'] for d in record['draws']])
            rows.append(dict(episode=s['episode'],rank=s['rank'],sku=s['sku'],day=s['day'],
                stock=s['obs'][s['sku']],forecast=s['obs'][-20+s['sku']],gains=s['gains'],
                economics={k:float(np.mean([e[k] for e in economic])) for k in TERMS}))
    assert seen==set(expected) and len(rows)==256
    groups={}
    for sku4 in [False,True]:
        for late in [False,True]:
            selected=[r for r in rows if (r['sku']==4)==sku4 and (r['day']==22)==late]
            label=f'sku4_{sku4}_day22_{late}'
            groups[label]=dict(**describe([r['gains'] for r in selected]),
                windows=len({r['episode'] for r in selected}),
                max_cover_count=sum(r['rank']==0 for r in selected),
                economics={k:float(np.mean([r['economics'][k] for r in selected])) for k in TERMS})
    result=dict(status='complete',groups=groups,rows=rows,new_simulator_days=0,policy_fits=0,
        hashes={f:digest(root/f) for f in ['policy.json','states.jsonl','verification.json']},script_sha256=digest(__file__),
        limitation='Four post-result descriptive partitions, no subgroup significance or actor improvement claim. Discounted suffix units, not reward/day. Full means are not split-selected economics.')
    out.mkdir(parents=True,exist_ok=False);write_json(out/'audit.json',result)
    print(json.dumps(groups),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.source,a.output)
