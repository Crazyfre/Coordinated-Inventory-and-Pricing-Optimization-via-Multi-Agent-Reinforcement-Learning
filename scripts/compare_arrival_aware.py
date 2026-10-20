"""One matched arrival-information feasibility comparison, no validation."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.stats import t
from scripts.arrival_collection import collect
from scripts.train_rollout_correction import digest
from scripts.diagnose_cover_search import CoverPolicy
from src.algos.rollout_correction import RolloutCorrection,fit_ridge
from src.algos.arrival_aware import arrival_features,controlled_legacy_features
from src.algos.structured_imitation import InventoryPositionLinearPolicy
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env


def dataset(records,n,h):
    xs={'legacy':[],'arrival':[]};ys=[];mask=[]
    for r in records:
        obs=np.asarray(r['state']['obs'],dtype=np.float32);ri=r['actions'].index(r['reference_action'])
        v=np.array([[b['value'] for b in d['branches']] for d in r['draws']]);g=v-v[:,ri,None]
        if r['day']>=27:np.testing.assert_array_equal(g,np.zeros_like(g))
        for j,a in enumerate(r['actions']):
            for label,fn in [('legacy',controlled_legacy_features),('arrival',arrival_features)]:
                xs[label].append(fn(obs,n,h,r['sku'],a-r['reference_action'],r['day']))
            ys.append(float(g[:,j].mean()));mask.append(j!=ri)
    return {k:np.asarray(v) for k,v in xs.items()},np.array(ys),np.array(mask)


def metrics(records,predictions,y,mask):
    results={};episode_errors={}
    for label,p in predictions.items():
        offset=0;errors=[]
        for r in records:
            end=offset+len(r['actions']);m=mask[offset:end]
            errors.append(float(np.mean((p[offset:end][m]-y[offset:end][m])**2)));offset=end
        strata=np.array(errors).reshape(16,5);episode=strata[:,:4].mean(1);episode_errors[label]=episode
        results[label]=dict(primary_mse=float(episode.mean()),stratum_mse=strata.mean(0).tolist(),episode_mse=episode.tolist(),state_mse=errors)
    gates={}
    for label in ['legacy','zero']:
        delta=episode_errors['arrival']-episode_errors[label];half=t.ppf(.975,15)*delta.std(ddof=1)/4
        gates[label]=dict(delta=float(delta.mean()),ci95=[float(delta.mean()-half),float(delta.mean()+half)],passed=bool(delta.mean()+half<0))
    return dict(models=results,gates=gates,diagnostic_gate=all(g['passed'] for g in gates.values()))


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();deadline=started+1200
    old=json.loads(Path('outputs/rollout_learning/ridge_v1/manifest.json').read_text())
    for k,p in old['paths'].items():assert digest(p)==old['hashes'][k]
    paths=dict(old['paths']);paths.update(runner=__file__,collector='scripts/arrival_collection.py',features='src/algos/arrival_aware.py',
        protocol='ARRIVAL_AWARE_PROTOCOL.md',resampling='scripts/audit_repeated_futures.py',rollout='scripts/audit_rollout_predictions.py',
        checkpoint='outputs/rollout_learning/ridge_v1/checkpoint.json')
    hashes={k:digest(p) for k,p in paths.items()};(out/'manifest.json').write_text(json.dumps(dict(paths=paths,hashes=hashes),indent=2))
    ck=json.loads(Path(paths['checkpoint']).read_text());torch.set_num_threads(1)
    env=make_research_env('S2',split='train',start=15,params=load_identifiable_training_calibration(paths['calibration']))
    center=InventoryPositionLinearPolicy.load(paths['center'],env.n,env.obs_pipe_horizon);target=CoverPolicy(env.n,env.obs_pipe_horizon,ck['cover'])
    model=RolloutCorrection(center,target,env.n,env.obs_pipe_horizon,ck['weights'],ck['scale'])
    counts={};models={};checkpoint_hashes={};trace_hashes={}
    for phase,seed0,reps,namespace in [('fit',990000,4,992100),('assessment',991000,8,993100)]:
        records=[];branches=days=probe_days=0
        for ep in range(16):
            data=collect(env,center,target,model,seed0+ep,1,reps,namespace,deadline,phase,episode_offset=ep)
            path=out/f'{phase}_{ep:03d}.json';path.write_text(json.dumps(data));trace_hashes[path.name]=digest(path)
            records.extend(data['records'])
            for r in data['records']:
                probe_days+=len(r['unresampled_probe'])
                for d in r['draws']:
                    branches+=len(d['branches']);days+=sum(len(b['trace']) for b in d['branches'])
            print(f'{phase} {ep+1}/16',flush=True)
        x,y,mask=dataset(records,env.n,env.obs_pipe_horizon)
        counts[phase]=dict(states=len(records),branches=branches,branch_days=days,probe_days=probe_days,replay_days=probe_days,original_days=448)
        if phase=='fit':
            for label in x:
                w,scale=fit_ridge(x[label],y);models[label]=dict(schema='arrival_comparison_ridge_v1',feature_schema=label,weights=w.tolist(),scale=scale.tolist(),n=env.n,horizon=env.obs_pipe_horizon)
                path=out/(label+'_checkpoint.json');path.write_text(json.dumps(models[label],indent=2));checkpoint_hashes[label]=digest(path)
            np.savez_compressed(out/'fit_data.npz',**x,y=y,mask=mask)
            (out/'frozen_fit.json').write_text(json.dumps(checkpoint_hashes))
        else:
            predictions={label:x[label]/np.array(m['scale'])@np.array(m['weights']) for label,m in models.items()};predictions['zero']=np.zeros(len(y))
            result=metrics(records,predictions,y,mask)
            np.savez_compressed(out/'assessment_data.npz',x_legacy=x['legacy'],x_arrival=x['arrival'],y=y,mask=mask,**predictions)
    assert {k:digest(p) for k,p in paths.items()}==hashes
    assert {label:digest(out/(label+'_checkpoint.json')) for label in models}==checkpoint_hashes
    (out/'trace_hashes.json').write_text(json.dumps(trace_hashes,indent=2))
    summary=dict(result=result,counts=counts,checkpoint_hashes=checkpoint_hashes,seconds=time.perf_counter()-started,fits=2,validation_episodes=0,held_out_test_used=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
