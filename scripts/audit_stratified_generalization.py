"""Leave-window-out prediction diagnosis; never exports a deployable policy."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.train_compact_residual import digest,write_json


def run(source,output):
    root=Path(source);out=Path(output)
    proof=json.loads((root/'verification.json').read_text());assert proof['status']=='passed'
    freeze=json.loads((root/'assessment_freeze.json').read_text())
    assert digest(root/'policy.json')==freeze['fit_hashes']['policy.json']
    p=json.loads((root/'policy.json').read_text());x=np.array(p['matrix']);y=np.array(p['targets'])
    samples=p['samples'];assert [(r['episode'],r['rank']) for r in samples]==[(e,j) for e in range(128) for j in [0,1]]
    full=x@np.array(p['weights']);held=np.zeros(256);constant=np.zeros(256)
    for ep in range(128):
        test=np.arange(256)//2==ep;train=~test
        w=np.linalg.lstsq(np.vstack([x[train],np.eye(37)]),np.r_[y[train],np.zeros(37)],rcond=None)[0]
        held[test]=x[test]@w;constant[test]=y[train].mean()
    stats={}
    for name,pred in [('in_sample',full),('leave_window_out',held),('leave_window_constant',constant)]:
        chosen=pred>0
        stats[name]=dict(mse=float(np.mean((pred-y)**2)),positive_predictions=int(chosen.sum()),
            positive_target_among_predicted=int(((y>0)&chosen).sum()),
            mean_target_when_predicted=float(y[chosen].mean()) if chosen.any() else None,
            mean_predicted_positive=float(pred[chosen].mean()) if chosen.any() else None)
    first=[];second=[];se=[]
    for r in samples:
        values=np.array(r['gains'])/p['base_prices'][r['sku']]
        first.append(values[:16].mean());second.append(values[16:].mean());se.append(values.std(ddof=1)/np.sqrt(32))
    result=dict(status='complete',prediction=stats,positive_targets=int((y>0).sum()),
        positive_in_both_halves=int(((np.array(first)>0)&(np.array(second)>0)).sum()),
        median_target_mc_se=float(np.median(se)),diagnostic_ridge_fits=128,new_simulator_days=0,
        deployed_policy_changed=False,
        limitation='Post-result cross-validation on enriched training states, not independent policy assessment. No hyperparameter selection or saved fold policies. Targets remain noisy; MSE and positive-label precision are not policy return.',
        hashes={f:digest(root/f) for f in ['policy.json','verification.json','assessment_freeze.json']},script_sha256=digest(__file__))
    out.mkdir(parents=True,exist_ok=False);write_json(out/'audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--source',required=True);a.add_argument('--output',required=True)
    args=a.parse_args();run(args.source,args.output)
