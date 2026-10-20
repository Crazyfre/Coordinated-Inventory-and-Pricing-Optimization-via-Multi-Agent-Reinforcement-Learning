"""Fixed matched ridge fits differing only in reward-target averaging."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.stats import t
from scripts.train_rollout_correction import digest
from scripts.audit_rollout_predictions import rollout
from scripts.audit_repeated_futures import snapshot,resample_future
from scripts.diagnose_counterfactual_rollouts import clone_environment,decision,advance,check_branches
from scripts.diagnose_cover_search import CoverPolicy
from src.algos.rollout_correction import RolloutCorrection,features,fit_ridge
from src.algos.structured_imitation import InventoryPositionLinearPolicy
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env


def matrices(records,n,h):
    xs=[];single=[];average=[];noise=[];mask=[]
    for r in records:
        obs=np.asarray(r['state']['obs'],dtype=np.float32);actions=r['actions'];ri=actions.index(r['reference_action'])
        v=np.array([[b['value'] for b in d['branches']] for d in r['draws']]);g=v-v[:,ri,None]
        for j,a in enumerate(actions):
            xs.append(features(obs,n,h,r['sku'],a-r['reference_action'],r['day']))
            single.append(g[0,j]);average.append(g[:,j].mean());noise.append(g[:,j].var(ddof=1)/len(g));mask.append(j!=ri)
    return np.array(xs),np.array(single),np.array(average),np.array(noise),np.array(mask)


def assess(records,predictions,means,noise,mask):
    models={};episode_errors={}
    for label,p in predictions.items():
        errors=[];regrets=[];best=[];offset=0
        for r in records:
            end=offset+len(r['actions']);m=mask[offset:end];y=means[offset:end];pr=p[offset:end]
            errors.append(float(np.mean((pr[m]-y[m])**2)))
            ranked=int(np.argmax(pr));regrets.append(float(y.max()-y[ranked]));best.append(bool(y[ranked]>=y.max()-1e-9));offset=end
        ep=np.array(errors).reshape(-1,2).mean(1);episode_errors[label]=ep
        models[label]=dict(episode_equal_mse=float(ep.mean()),action_pooled_mse=float(np.mean((p[mask]-means[mask])**2)),
                           episode_mse=ep.tolist(),state_mse=errors,mean_rank_regret=float(np.mean(regrets)),tied_best_rate=float(np.mean(best)))
    delta=episode_errors['averaged']-episode_errors['single'];half=t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))
    return dict(models=models,mean_target_noise=float(noise[mask].mean()),nonreference_actions=int(mask.sum()),
                averaged_minus_single_mse=float(delta.mean()),difference_ci95=[float(delta.mean()-half),float(delta.mean()+half)],
                prediction_gate=bool(delta.mean()+half<0))


def collect(env,center,target,model,seed0,count,reps,namespace,deadline,label):
    records=[];originals=[]
    for ep,seed in enumerate(range(seed0,seed0+count)):
        start=int(np.random.default_rng(np.random.SeedSequence([seed,314159])).integers(15,1859))
        env.sim.start_day_index=start;env.reset(seed=seed);original=[];episode=[]
        while not env._finished:
            c,ref=decision(env,center,target);day=env.day;chosen=model.action(env.replenishment_observation(),day)
            if day in (8,15):
                changed=np.flatnonzero(chosen!=ref);assert len(changed)<=1
                sku=int(changed[0]) if len(changed) else (ep*2+(day-8)//7)%env.n
                actions=list(range(max(0,int(c[sku])-2),min(40,int(c[sku])+2)+1))
                before=snapshot(env);keys_before=[env.sim.scenario_seed,env.pipeline.scenario_seed]
                probe=rollout(clone_environment(env),center,target,model,chosen,'learned',deadline);draws=[]
                for rep in range(reps):
                    keys=[int(np.random.SeedSequence([namespace,ep,day,rep,s]).generate_state(1,dtype=np.uint64)[0]) for s in (0,1)]
                    bs=[]
                    for a in actions:
                        first=ref.copy();first[sku]=a
                        trace=rollout(resample_future(env,keys),center,target,model,first,'reference',deadline)
                        bs.append(dict(action=a,trace=trace,value=float(np.dot(.99**np.arange(len(trace)),[row['reward'] for row in trace]))))
                    baseline=next(b for b in bs if b['action']==ref[sku]);check_branches(bs,baseline['trace'])
                    if rep==0:
                        replay=rollout(resample_future(env,keys),center,target,model,ref,'reference',deadline)
                        assert replay==baseline['trace']
                    draws.append(dict(replicate=rep,keys=keys,branches=bs))
                assert snapshot(env)==before and [env.sim.scenario_seed,env.pipeline.scenario_seed]==keys_before
                episode.append(dict(seed=seed,start=start,day=day,sku=sku,actions=actions,reference_action=int(ref[sku]),
                    chosen_action=int(chosen[sku]),state=before,draws=draws,unresampled_probe=probe,replicate0_replay=replay))
            row,_=advance(env,chosen);original.append(row)
        for r in episode:assert r['unresampled_probe']==original[r['day']-1:]
        records.extend(episode);originals.append(dict(seed=seed,start=start,trace=original))
        if (ep+1)%4==0:print(f'{label}: {ep+1}/{count} scenarios',flush=True)
    return dict(records=records,originals=originals)


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();deadline=started+600
    old=json.loads(Path('outputs/rollout_learning/ridge_v1/manifest.json').read_text())
    for k,p in old['paths'].items():assert digest(p)==old['hashes'][k]
    paths=dict(old['paths']);paths.update(comparison=__file__,comparison_protocol='AVERAGED_TARGET_PROTOCOL.md',
        resampling='scripts/audit_repeated_futures.py',rollout='scripts/audit_rollout_predictions.py',
        checkpoint='outputs/rollout_learning/ridge_v1/checkpoint.json')
    hashes={k:digest(p) for k,p in paths.items()};(out/'manifest.json').write_text(json.dumps(dict(paths=paths,hashes=hashes),indent=2))
    ck=json.loads(Path(paths['checkpoint']).read_text());torch.set_num_threads(1)
    env=make_research_env('S2',split='train',start=15,params=load_identifiable_training_calibration(paths['calibration']))
    center=InventoryPositionLinearPolicy.load(paths['center'],env.n,env.obs_pipe_horizon);target=CoverPolicy(env.n,env.obs_pipe_horizon,ck['cover'])
    model=RolloutCorrection(center,target,env.n,env.obs_pipe_horizon,ck['weights'],ck['scale'])
    fit=collect(env,center,target,model,970000,16,8,972100,deadline,'fit')
    (out/'fit_traces.json').write_text(json.dumps(fit))
    x,single,average,_,_=matrices(fit['records'],env.n,env.obs_pipe_horizon);models={}
    for label,y in [('single',single),('averaged',average)]:
        w,scale=fit_ridge(x,y)
        models[label]=dict(schema='averaged_target_ridge_diagnostic_v1',weights=w.tolist(),scale=scale.tolist(),
                           n=env.n,horizon=env.obs_pipe_horizon,reference_checkpoint_sha256=hashes['checkpoint'],future_count=1 if label=='single' else 8)
        (out/(label+'_checkpoint.json')).write_text(json.dumps(models[label],indent=2))
    checkpoint_hashes={label:digest(out/(label+'_checkpoint.json')) for label in models}
    np.savez_compressed(out/'fit_data.npz',x=x,single=single,averaged=average)
    print('Both fits frozen before independent assessment',flush=True)
    assessment=collect(env,center,target,model,971000,8,16,973100,deadline,'assessment')
    (out/'assessment_traces.json').write_text(json.dumps(assessment))
    ex,_,ey,noise,mask=matrices(assessment['records'],env.n,env.obs_pipe_horizon)
    predictions={label:ex/np.asarray(m['scale'])@np.asarray(m['weights']) for label,m in models.items()};predictions['zero']=np.zeros(len(ey))
    np.savez_compressed(out/'assessment_data.npz',x=ex,y=ey,noise=noise,mask=mask,**predictions)
    result=assess(assessment['records'],predictions,ey,noise,mask)
    counts={label:dict(states=len(data['records']),branches=sum(len(d['branches']) for r in data['records'] for d in r['draws']),
                      branch_days=sum(len(b['trace']) for r in data['records'] for d in r['draws'] for b in d['branches'])) for label,data in [('fit',fit),('assessment',assessment)]}
    assert {k:digest(p) for k,p in paths.items()}==hashes
    assert {label:digest(out/(label+'_checkpoint.json')) for label in models}==checkpoint_hashes
    summary=dict(result=result,counts=counts,checkpoint_hashes=checkpoint_hashes,original_days=672,probe_days=840,replay_days=840,
                 seconds=time.perf_counter()-started,fits=2,validation_episodes=0,held_out_test_used=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
