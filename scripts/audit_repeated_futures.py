"""Fixed-state future resampling without resetting inventory or pipeline."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.stats import t
from scripts.train_rollout_correction import digest
from scripts.audit_rollout_predictions import rollout
from scripts.diagnose_counterfactual_rollouts import clone_environment,decision,advance,check_branches
from scripts.diagnose_cover_search import CoverPolicy
from src.algos.rollout_correction import RolloutCorrection,features
from src.algos.structured_imitation import InventoryPositionLinearPolicy
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env


def snapshot(env):
    return dict(day=int(env.day),obs=env.replenishment_observation().tolist(),
                inventory=env.inventory.on_hand.tolist(),history=[a.tolist() for a in env._demand_history],
                pipeline={str(k):v.tolist() for k,v in sorted(env.pipeline._in_flight.items())},
                arrivals=env._arrivals_today.tolist(),accepted=env._accepted_today.tolist(),
                clipped=env._clipped_today.tolist(),price=env._current_price.tolist(),
                pending=None if env._pending_price is None else env._pending_price.tolist(),
                start=int(env.sim.start_day_index))


def future_keys(ep,day,rep):
    return [int(np.random.SeedSequence([960100,ep,day,rep,stream]).generate_state(1,dtype=np.uint64)[0]) for stream in (0,1)]


def resample_future(env,keys):
    before=snapshot(env)
    clone=clone_environment(env)
    clone.sim.scenario_seed,clone.pipeline.scenario_seed=map(int,keys)
    assert clone.sim is clone.demand_source.sim and clone.sim is clone.forecaster.sim
    assert clone.sim is not env.sim and clone.pipeline is not env.pipeline
    assert snapshot(clone)==before and snapshot(env)==before
    return clone


def summarize(records):
    per_state=[];predictions=[];outcomes=[]
    for r in records:
        p=np.array(r['predictions']);a=r['actions'];ref=a.index(r['reference_action'])
        values=np.array([[b['value'] for b in draw['branches']] for draw in r['draws']])
        gains=values-values[:,ref,None]
        means=gains.mean(0);variance=gains.var(0,ddof=1);se=np.sqrt(variance/16)
        mask=np.array(a)!=r['reference_action'];predictions.extend(p[mask]);outcomes.extend(gains[:,mask].T)
        chosen=a.index(r['chosen_action']);half=float(t.ppf(.975,15)*se[chosen])
        ranked=int(np.argmax(p))
        per_state.append(dict(seed=r['seed'],day=r['day'],sku=r['sku'],actions=a,predictions=p.tolist(),
            mean_gain=means.tolist(),variance=variance.tolist(),mc_se=se.tolist(),
            chosen_gain=float(means[chosen]),chosen_ci95=[float(means[chosen]-half),float(means[chosen]+half)],
            mean_rank_regret=float(means.max()-means[ranked]),mean_rank_tied_best=bool(means[ranked]>=means.max()-1e-9)))
    p=np.array(predictions);y=np.array(outcomes);means=y.mean(1);var=y.var(1,ddof=1)
    mean_noise=float(var.mean());stats={}
    for label,pred in [('model',p),('zero',np.zeros_like(p))]:
        mse=float(np.mean((pred-means)**2));single=float(np.mean((pred[:,None]-y)**2))
        stats[label]=dict(single_future_mse=single,mean_target_mse=mse,
                         persistent_mse_estimate=mse-mean_noise/16)
        np.testing.assert_allclose(single,mse+15/16*mean_noise,rtol=1e-12,atol=1e-12)
    return dict(states=per_state,nonreference_actions=len(p),mean_within_state_action_variance=mean_noise,
                mean_target_noise_variance=mean_noise/16,errors=stats,
                mean_rank_regret=float(np.mean([s['mean_rank_regret'] for s in per_state])),
                mean_rank_tied_best_rate=float(np.mean([s['mean_rank_tied_best'] for s in per_state])))


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();deadline=started+600
    old=json.loads(Path('outputs/rollout_learning/ridge_v1/manifest.json').read_text())
    for k,p in old['paths'].items():assert digest(p)==old['hashes'][k]
    paths=dict(old['paths']);paths.update(audit=__file__,audit_protocol='REPEATED_FUTURE_PROTOCOL.md',
        rollout='scripts/audit_rollout_predictions.py',checkpoint='outputs/rollout_learning/ridge_v1/checkpoint.json')
    hashes={k:digest(p) for k,p in paths.items()};(out/'manifest.json').write_text(json.dumps(dict(paths=paths,hashes=hashes),indent=2))
    ck=json.loads(Path(paths['checkpoint']).read_text());torch.set_num_threads(1)
    env=make_research_env('S2',split='train',start=15,params=load_identifiable_training_calibration(paths['calibration']))
    center=InventoryPositionLinearPolicy.load(paths['center'],env.n,env.obs_pipe_horizon)
    target=CoverPolicy(env.n,env.obs_pipe_horizon,ck['cover'])
    model=RolloutCorrection(center,target,env.n,env.obs_pipe_horizon,ck['weights'],ck['scale'])
    records=[];originals=[]
    for ep,seed in enumerate(range(960000,960004)):
        start=int(np.random.default_rng(np.random.SeedSequence([seed,314159])).integers(15,1859))
        env.sim.start_day_index=start;env.reset(seed=seed);original=[];episode=[]
        while not env._finished:
            c,ref=decision(env,center,target);obs=env.replenishment_observation();day=env.day;chosen=model.action(obs,day)
            if day in (8,15):
                changed=np.flatnonzero(chosen!=ref);assert len(changed)<=1
                sku=int(changed[0]) if len(changed) else (ep*2+(day-8)//7)%env.n
                actions=list(range(max(0,int(c[sku])-2),min(40,int(c[sku])+2)+1))
                before=snapshot(env);original_keys=[env.sim.scenario_seed,env.pipeline.scenario_seed]
                probe=rollout(clone_environment(env),center,target,model,chosen,'learned',deadline)
                draws=[]
                for rep in range(16):
                    keys=future_keys(ep,day,rep);bs=[]
                    for a in actions:
                        first=ref.copy();first[sku]=a
                        branch=resample_future(env,keys)
                        trace=rollout(branch,center,target,model,first,'reference',deadline)
                        bs.append(dict(action=a,trace=trace,value=float(np.dot(.99**np.arange(len(trace)),[row['reward'] for row in trace]))))
                    baseline=next(b for b in bs if b['action']==int(ref[sku]));check_branches(bs,baseline['trace'])
                    if rep==0:
                        replay=rollout(resample_future(env,keys),center,target,model,ref,'reference',deadline)
                        assert replay==baseline['trace']
                    draws.append(dict(replicate=rep,keys=keys,branches=bs))
                assert snapshot(env)==before and [env.sim.scenario_seed,env.pipeline.scenario_seed]==original_keys
                demands={json.dumps([row['demand'] for row in d['branches'][0]['trace']]) for d in draws}
                assert len(demands)>1
                predictions=[float((features(obs,env.n,env.obs_pipe_horizon,sku,a-int(ref[sku]),day)/model.scale)@model.weights) for a in actions]
                episode.append(dict(seed=seed,start=start,day=day,sku=sku,actions=actions,reference_action=int(ref[sku]),
                    chosen_action=int(chosen[sku]),predictions=predictions,state=before,original_keys=original_keys,
                    draws=draws,unresampled_probe=probe,replicate0_replay=replay))
                print(f'Completed fixed state {ep*2+(day-8)//7+1}/8',flush=True)
            row,_=advance(env,chosen);original.append(row)
        for r in episode:assert r['unresampled_probe']==original[r['day']-1:]
        records.extend(episode);originals.append(dict(seed=seed,start=start,trace=original))
    assert {k:digest(p) for k,p in paths.items()}==hashes
    (out/'traces.json').write_text(json.dumps(dict(records=records,originals=originals)))
    summary=dict(statistics=summarize(records),branch_days=sum(len(b['trace']) for r in records for d in r['draws'] for b in d['branches']),
                 branches=sum(len(d['branches']) for r in records for d in r['draws']),original_days=112,probe_days=140,replay_days=140,
                 seconds=time.perf_counter()-started,fits=0,validation_episodes=0,held_out_test_used=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
