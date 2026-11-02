"""Prospective rollout-rule fitting followed by frozen complete-episode assessment."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.algos.rollout_pricing_rule import act,fit
from src.algos.parameter_policy_gradient import order_action
from src.env.interface import JointAction
from src.experiments.research import make_research_env
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from scripts.diagnose_conditional_pricing import advance,suffix,STUDY
from scripts.diagnose_counterfactual_rollouts import clone_environment
from scripts.audit_repeated_futures import snapshot,resample_future
from scripts.train_compact_residual import INPUTS,digest,write_json,serial


def assess(rows):
    labels=['learned','baseline','heuristic'];keys=list(range(2490000,2490064))
    index={(r['label'],r['key']):r for r in rows}
    if len(index)!=len(rows) or set(index)!={(l,k) for l in labels for k in keys}:
        raise ValueError('complete unique paired assessment grid required')
    means={l:{m:float(np.mean([index[l,k][m] for k in keys])) for m in ['reward_day','fill']} for l in labels}
    draws=np.random.default_rng(20261103).integers(64,size=(10000,64));effects={}
    for ref in ['baseline','heuristic']:
        for metric in ['reward_day','fill']:
            x=np.array([index['learned',k][metric]-index[ref,k][metric] for k in keys])
            effects[f'{ref}:{metric}']=dict(mean=float(x.mean()),ci9875=np.quantile(x[draws].mean(1),[.00625,.99375]).tolist())
    primary=effects['baseline:reward_day'];service=effects['baseline:fill']
    matched=effects['heuristic:reward_day'];matchedservice=effects['heuristic:fill']
    return dict(means=means,effects=effects,
        primary_pass=bool(primary['mean']>=.5 and primary['ci9875'][0]>0 and service['ci9875'][0]>=-.005),
        heuristic_pass=bool(matched['ci9875'][0]>0 and matchedservice['ci9875'][0]>=-.005),
        price_counts={l:np.sum([index[l,k]['price_counts'] for k in keys],axis=0).tolist() for l in labels})


def collect(env,covers,key,label,thresholds,counters,deadline):
    start=int(np.random.default_rng(np.random.SeedSequence([key,314159])).integers(15,1859))
    env.sim.start_day_index=start;env.reset(seed=key);trace=[]
    for day in range(1,29):
        if time.perf_counter()>deadline:raise TimeoutError('assessment cap')
        pobs=env.pricing_observation().copy() if env.pricing_due else None
        pa=act(pobs,thresholds) if pobs is not None else None
        if pa is not None:env.select_price(pa)
        obs=env.replenishment_observation().copy();oa=order_action(obs,day,covers)
        _,reward,done,info=env.step(JointAction(None,oa));assert done==(day==28)
        trace.append(serial(dict(day=day,start=start,obs=obs,actions=oa,pricing_obs=pobs,price_actions=pa,
            reward=reward,info=info,demand_seed=int(env.sim.scenario_seed),supplier_seed=int(env.pipeline.scenario_seed))))
        counters['days']+=1
    demand=sum(sum(r['info']['demand']) for r in trace);sales=sum(sum(r['info']['sales']) for r in trace)
    row=dict(label=label,key=key,reward_day=sum(r['reward'] for r in trace)/28,
        fill=sales/demand if demand else 1.,price_counts=np.bincount(np.concatenate(
            [r['price_actions'] for r in trace if r['price_actions'] is not None]),minlength=5).tolist())
    return row,trace


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    prior=Path('outputs/structured_mappo/pricing_candidates_v1/verification.json')
    assert json.loads(prior.read_text())['positive_states']==1
    covers=np.array(json.loads((STUDY/'sku_seed211_final.json').read_text())['mean'])
    paths={str(p) for folder in ['src','scripts'] for p in Path(folder).rglob('*.py')}
    paths|={str(p) for p in Path('data').rglob('*') if p.is_file() and ('aligned' in str(p) or 'sku' in p.name.lower())}
    paths|={'src/config/skus.json',INPUTS['calibration'],str(STUDY/'sku_seed211_final.json'),
        'PRICING_ROLLOUT_RULE_PROTOCOL.md','PRICING_ROLLOUT_RULE_V2_PROTOCOL.md',str(prior),'outputs/structured_mappo/rollout_pricing_timeout_audit_v1/audit.json','outputs/structured_mappo/rollout_archive_profile_v1/profile.json'}
    hashes={p:digest(p) for p in sorted(paths)}
    write_json(out/'manifest.json',dict(hashes=hashes,covers=covers,planned_days=206388,fit_cap=3600,assessment_cap=600,total_cap=4200))
    torch.set_num_threads(1);began=time.perf_counter();counters=dict(days=0);stage='fit';fitrows=[]
    try:
        env=make_research_env('S2',split='train',start=15,pricing_scope='per_sku',
            params=load_identifiable_training_calibration(INPUTS['calibration']))
        with (out/'fit_states.jsonl').open('w') as states,(out/'originals.jsonl').open('w') as originals:
            for ep,key in enumerate(range(2470000,2470016)):
                env.sim.start_day_index=int(np.random.default_rng(np.random.SeedSequence([key,314159])).integers(15,1859))
                env.reset(seed=key);original=[];probes={}
                while not env._finished:
                    day=int(env.day)
                    if env.pricing_due:
                        before=snapshot(env);obs=env.pricing_observation().tolist()
                        probe=suffix(clone_environment(env),covers,0,4,counters,began+3600);probes[day]=probe
                        draws=[];advantages=[];replays=[];replay_sku=(ep*4+(day-1)//7)%10
                        for rep in range(16):
                            keys=[int(np.random.SeedSequence([2480000,ep,day,rep,s]).generate_state(1,dtype=np.uint64)[0]) for s in [0,1]]
                            branches=[]
                            for sku in [-1]+list(range(10)):
                                branch=resample_future(env,keys);initial=snapshot(branch)
                                trace=suffix(branch,covers,max(sku,0),4 if sku==-1 else 3,counters,began+3600)
                                value=sum(.99**i*r['reward'] for i,r in enumerate(trace))
                                branches.append(dict(sku=sku,initial=initial,trace=trace,value=value))
                                if rep==0 and sku in [-1,replay_sku]:
                                    replay=suffix(resample_future(env,keys),covers,max(sku,0),4 if sku==-1 else 3,counters,began+3600)
                                    assert replay==trace;replays.append(dict(sku=sku,trace=replay))
                            advantages.append([b['value']-branches[0]['value'] for b in branches[1:]])
                            draws.append(dict(keys=keys,branches=branches))
                        assert snapshot(env)==before
                        row=dict(episode=ep,day=day,obs=obs,advantages=advantages);fitrows.append(row)
                        states.write(json.dumps(dict(episode=ep,key=key,day=day,before=before,after=snapshot(env),
                            probe=probe,draws=draws,replays=replays,statistics=row),allow_nan=False)+'\n');states.flush()
                    r,_=advance(env,covers,day,0,4,counters,began+3600);original.append(r)
                for day,probe in probes.items():assert probe==original[day-1:]
                originals.write(json.dumps(dict(episode=ep,key=key,trace=original),allow_nan=False)+'\n');originals.flush()
                write_json(out/'progress.json',dict(stage=stage,windows=ep+1,days=counters['days'],elapsed_seconds=time.perf_counter()-began))
                print(f'Pricing rule fit: {ep+1}/16 windows',flush=True)
        assert counters['days']==200928
        fitted=fit(fitrows);write_json(out/'policy.json',fitted)
        fit_seconds=time.perf_counter()-began;assert fit_seconds<=3600
        assert all(digest(p)==h for p,h in hashes.items())
        fit_hashes={p:digest(out/p) for p in ['policy.json','fit_states.jsonl','originals.jsonl']}
        write_json(out/'assessment_freeze.json',dict(fit_hashes=fit_hashes,source_hashes=hashes,days=counters['days'],
            fit_seconds=fit_seconds,thresholds=fitted['thresholds']))
        freeze_hash=digest(out/'assessment_freeze.json');stage='assessment';assessment_start=time.perf_counter()
        controllers={'learned':fitted['thresholds'],'baseline':[None]*10,'heuristic':[1.5]*10};rows=[];first={}
        with (out/'assessment.jsonl').open('w') as handle:
            for key in range(2490000,2490064):
                for label,thresholds in controllers.items():
                    row,trace=collect(env,covers,key,label,thresholds,counters,min(began+4200,assessment_start+600));rows.append(row)
                    handle.write(json.dumps(dict(label=label,key=key,trace=trace,result=row),allow_nan=False)+'\n')
                    if key==2490000:first[label]=trace
                handle.flush();write_json(out/'progress.json',dict(stage=stage,scenarios=key-2490000+1,days=counters['days']))
        with (out/'replays.jsonl').open('w') as handle:
            for label,thresholds in controllers.items():
                _,trace=collect(env,covers,2490000,label,thresholds,counters,min(began+4200,assessment_start+600))
                assert trace==first[label];handle.write(json.dumps(dict(label=label,key=2490000,trace=trace),allow_nan=False)+'\n')
        assessment_seconds=time.perf_counter()-assessment_start;elapsed=time.perf_counter()-began
        assert counters['days']==206388 and assessment_seconds<=600 and elapsed<=4200
        assert all(digest(p)==h for p,h in hashes.items()) and all(digest(out/p)==h for p,h in fit_hashes.items())
        assert digest(out/'assessment_freeze.json')==freeze_hash
        analysis=assess(rows)
        write_json(out/'summary.json',dict(status='complete',analysis=analysis,rows=rows,days=counters['days'],
            fit_seconds=fit_seconds,assessment_seconds=assessment_seconds,seconds=elapsed,test_scoring=False,
            freeze_hash=freeze_hash,trace_hashes={p.name:digest(p) for p in out.glob('*.jsonl')}))
        print(json.dumps(analysis),flush=True)
    except Exception as exc:
        write_json(out/'failed_status.json',dict(stage=stage,error=str(exc),counters=counters));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)

