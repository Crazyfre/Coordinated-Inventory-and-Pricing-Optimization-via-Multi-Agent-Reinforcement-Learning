"""Frozen training-only prediction/continuation assessment."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.stats import t
from scripts.train_rollout_correction import digest
from scripts.diagnose_counterfactual_rollouts import clone_environment,decision,advance,check_branches
from scripts.diagnose_cover_search import CoverPolicy
from src.algos.rollout_correction import RolloutCorrection,features
from src.algos.structured_imitation import InventoryPositionLinearPolicy
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env


def interval(values):
    a=np.asarray(values,dtype=float)
    width=float(t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)))
    return dict(mean=float(a.mean()),ci95=[float(a.mean()-width),float(a.mean()+width)])


def metrics(records):
    result={}
    for source in ['reference','learned']:
        rs=[r for r in records if r['source']==source]
        source_result={};gains_by_cont={}
        for continuation in ['reference','learned']:
            predicted=[];actual=[];regrets=[];best=[];gains=[];active=[]
            for r in rs:
                bs=r['branches'][continuation]
                ref=next(b for b in bs if b['action']==r['reference_action'])
                ys=np.array([b['value']-ref['value'] for b in bs])
                ps=np.array(r['predictions'])
                mask=np.array([b['action']!=r['reference_action'] for b in bs])
                predicted.extend(ps[mask]);actual.extend(ys[mask])
                chosen=next(i for i,b in enumerate(bs) if b['action']==r['chosen_action'])
                ranked=int(np.argmax(ps))
                regrets.append(float(ys.max()-ys[ranked]));best.append(bool(ys[ranked]>=ys.max()-1e-9))
                gains.append(float(ys[chosen]));active.append(r['chosen_action']!=r['reference_action'])
            p,y=np.array(predicted),np.array(actual)
            scenario_gains=np.array(gains).reshape(16,4).mean(1)
            gains_by_cont[continuation]=scenario_gains
            source_result[continuation]=dict(nonreference_actions=len(y),rmse=float(np.sqrt(np.mean((p-y)**2))),
                zero_rmse=float(np.sqrt(np.mean(y*y))),bias=float(np.mean(p-y)),
                correlation=float(np.corrcoef(p,y)[0,1]) if p.std()>0 and y.std()>0 else None,
                mean_realized_ranking_regret=float(np.mean(regrets)),realized_tied_best_rate=float(np.mean(best)),
                active_states=int(sum(active)),chosen_gain=interval(scenario_gains),scenario_mean_chosen_gain=scenario_gains.tolist())
        source_result['continuation_delta']=interval(gains_by_cont['learned']-gains_by_cont['reference'])
        result[source]=source_result
    return result


def rollout(env,center,target,model,first,continuation,deadline):
    rows=[]
    while not env._finished:
        if time.perf_counter()>deadline:raise TimeoutError('600-second prediction audit budget')
        _,ref=decision(env,center,target)
        action=ref if continuation=='reference' else model.action(env.replenishment_observation(),env.day)
        row,_=advance(env,first if not rows else action);rows.append(row)
    return rows


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();deadline=started+600
    old=json.loads(Path('outputs/rollout_learning/ridge_v1/manifest.json').read_text())
    for key,path in old['paths'].items():assert digest(path)==old['hashes'][key]
    paths=dict(old['paths']);paths.update(audit=__file__,audit_protocol='ROLLOUT_PREDICTION_PROTOCOL.md',
                                         checkpoint='outputs/rollout_learning/ridge_v1/checkpoint.json')
    hashes={k:digest(p) for k,p in paths.items()}
    (out/'manifest.json').write_text(json.dumps(dict(paths=paths,hashes=hashes),indent=2))
    ck=json.loads(Path(paths['checkpoint']).read_text());torch.set_num_threads(1)
    env=make_research_env('S2',split='train',start=15,params=load_identifiable_training_calibration(paths['calibration']))
    center=InventoryPositionLinearPolicy.load(paths['center'],env.n,env.obs_pipe_horizon)
    target=CoverPolicy(env.n,env.obs_pipe_horizon,ck['cover'])
    model=RolloutCorrection(center,target,env.n,env.obs_pipe_horizon,ck['weights'],ck['scale'])
    records=[];originals=[]
    for source in ['reference','learned']:
        for ep,seed in enumerate(range(950000,950016)):
            start=int(np.random.default_rng(np.random.SeedSequence([seed,314159])).integers(15,1859))
            env.sim.start_day_index=start;env.reset(seed=seed);episode=[];original=[]
            while not env._finished:
                c,ref=decision(env,center,target);day=env.day;obs=env.replenishment_observation()
                chosen=model.action(obs,day)
                if day in (1,8,15,22):
                    changed=np.flatnonzero(chosen!=ref)
                    assert len(changed)<=1
                    sku=int(changed[0]) if len(changed) else (ep*4+(day-1)//7)%env.n
                    candidates=list(range(max(0,int(c[sku])-2),min(40,int(c[sku])+2)+1))
                    predictions=[float((features(obs,env.n,env.obs_pipe_horizon,sku,a-int(ref[sku]),day)/model.scale)@model.weights) for a in candidates]
                    branches={}
                    for continuation in ['reference','learned']:
                        bs=[]
                        for a in candidates:
                            first=ref.copy();first[sku]=a
                            trace=rollout(clone_environment(env),center,target,model,first,continuation,deadline)
                            bs.append(dict(action=a,trace=trace,value=float(np.dot(.99**np.arange(len(trace)),[r['reward'] for r in trace]))))
                        reference=next(b for b in bs if b['action']==ref[sku])
                        check_branches(bs,reference['trace']);branches[continuation]=bs
                    episode.append(dict(source=source,seed=seed,start=start,day=day,sku=sku,obs=obs.tolist(),
                                        center=int(c[sku]),reference_action=int(ref[sku]),chosen_action=int(chosen[sku]),
                                        predictions=predictions,branches=branches))
                row,_=advance(env,ref if source=='reference' else chosen);original.append(row)
            for r in episode:
                first=r['reference_action'] if source=='reference' else r['chosen_action']
                b=next(b for b in r['branches'][source] if b['action']==first)
                assert b['trace']==original[r['day']-1:]
            records.extend(episode);originals.append(dict(source=source,seed=seed,start=start,trace=original))
            if (ep+1)%4==0:print(f'{source} states: {ep+1}/16 scenarios',flush=True)
    assert [[r['demand'] for r in e['trace']] for e in originals[:16]]==[[r['demand'] for r in e['trace']] for e in originals[16:]]
    assert {k:digest(p) for k,p in paths.items()}==hashes
    (out/'traces.json').write_text(json.dumps(dict(records=records,originals=originals)))
    summary=dict(metrics=metrics(records),states=len(records),branches=sum(len(bs) for r in records for bs in r['branches'].values()),
                 branch_days=sum(len(b['trace']) for r in records for bs in r['branches'].values() for b in bs),
                 original_days=896,seconds=time.perf_counter()-started,fits=0,validation_episodes=0,held_out_test_used=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
