"""Complete fixed20-state descriptive feature audit; no fit or simulation."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.algos.cross_sku_pricing_features import features,NAMES
from scripts.train_compact_residual import digest,write_json


def run(output):
    original=Path('outputs/structured_mappo/stratified_pricing_v1');root=Path('outputs/structured_mappo/pricing_magnitude_v1')
    assert json.loads((root/'verification.json').read_text())['status']=='passed'
    manifest=json.loads((root/'manifest.json').read_text())
    assert digest(original/'policy.json')==manifest['hashes'][str(original/'policy.json')]
    p=json.loads((original/'policy.json').read_text());s=json.loads((root/'summary.json').read_text())
    samples={r['episode']:r for r in p['samples'] if r['sku']==4 and r['day']==22}
    rows=[]
    for state in s['analysis']['states']:
        r=samples[state['episode']];x=np.array(r['obs']);b=np.array(p['base_prices']);f=features(x,b)[4]
        # Independent direct per-SKU loop reconstructs all four quantities.
        deficit=sum(max(7*x[-20+j]-x[j],0) for j in range(10) if j!=4)
        value=sum(b[j]*max(7*x[-20+j]-x[j],0) for j in range(10) if j!=4)
        excess=sum(max(x[j]-7*x[-20+j],0) for j in range(10) if j!=4)
        expected=[deficit/500,value/(500*b.mean()),excess/500,
            max(x[4]-7*x[-16],0)/500*deficit/500*(1-np.clip(x[-6]/500,0,1))]
        np.testing.assert_allclose(f,expected,rtol=0,atol=1e-14)
        rows.append(dict(episode=r['episode'],features=dict(zip(NAMES,f.tolist())),
            stock=float(x[4]),forecast=float(x[-16]),headroom=float(x[-6]),
            index0_gain=state['action_mean'][0]))
    assert len(rows)==len(samples)==20
    y=np.array([r['index0_gain'] for r in rows]);correlations={}
    for name in NAMES:
        v=np.array([r['features'][name] for r in rows])
        correlations[name]=float(np.corrcoef(v,y)[0,1]) if np.std(v)>0 else None
    result=dict(status='complete',rows=rows,correlations=correlations,new_simulator_days=0,policy_fits=0,
        limitation='Post-result descriptive correlations in20 enriched states; no generalization or causal claim.',
        hashes={str(path):digest(path) for path in [original/'policy.json',root/'summary.json',root/'verification.json',
            Path('CROSS_SKU_FEATURE_SPEC.md'),Path('src/algos/cross_sku_pricing_features.py'),Path(__file__)]})
    out=Path(output);out.mkdir(parents=True,exist_ok=False);write_json(out/'audit.json',result)
    print(json.dumps(dict(correlations=correlations,rows=rows)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
