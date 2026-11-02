"""Prospective enriched-state rollout regression with two explicit freezes."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.algos.cross_sku_rollout import selection as select_candidates, matrix as features, fit, act
from src.algos.rollout_pricing_rule import act as threshold_act
from src.algos.parameter_policy_gradient import order_action
from src.env.interface import JointAction
from src.experiments.research import make_research_env
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from scripts.diagnose_conditional_pricing import advance,suffix,STUDY
from scripts.diagnose_counterfactual_rollouts import clone_environment
from scripts.audit_repeated_futures import snapshot,resample_future
from scripts.train_compact_residual import INPUTS,digest,write_json,serial


def reset(env,key):
    env.sim.start_day_index=int(np.random.default_rng(np.random.SeedSequence([key,314159])).integers(15,1859))
    env.reset(seed=key)


def freeze_selection(out,selections,hashes,counters,base_prices):
    if len(selections)!=128 or [r['episode'] for r in selections]!=list(range(128)):
        raise ValueError('all128 windows required before any branching')
    length=0
    for r in selections:
        if r['key']!=2540000+r['episode']:raise ValueError('declared scan key required')
        expected=[list(v) for v in select_candidates({int(k):v for k,v in r['observations'].items()},r['episode'],base_prices)]
        if r['selected']!=expected:raise ValueError('selection differs from observable-only rule')
        length+=sum(29-day for day,_ in expected)
    if counters['days']!=3584:raise ValueError('exact128 scan episodes required')
    if not all(digest(p)==h for p,h in hashes.items()):raise ValueError('source changed during scan')
    result=dict(selections=selections,suffix_length_sum=length,planned_days=21616+67*length,
        scan_sha256=digest(out/'scan.jsonl'),source_hashes=hashes,days=counters['days'])
    write_json(out/'selection_freeze.json',result)
    return result


def assess(rows):
    keys=list(range(2560000,2560128));labels=['cross','local','baseline','heuristic']
    index={(r['label'],r['key']):r for r in rows}
    if len(index)!=len(rows) or set(index)!={(l,k) for l in labels for k in keys}:
        raise ValueError('complete unique assessment grid required')
    resamples=np.random.default_rng(20261107).integers(128,size=(10000,128));effects={}
    for ref in ['baseline','local','heuristic']:
        for metric in ['reward_day','fill']:
            x=np.array([index['cross',k][metric]-index[ref,k][metric] for k in keys])
            effects[f'{ref}:{metric}']=dict(mean=float(x.mean()),ci991667=np.quantile(x[resamples].mean(1),[.05/12,1-.05/12]).tolist())
    p=effects['baseline:reward_day'];f=effects['baseline:fill']
    return dict(effects=effects,primary_pass=bool(p['mean']>=.5 and p['ci991667'][0]>0 and f['ci991667'][0]>=-.005),
        representation_pass=bool(effects['local:reward_day']['ci991667'][0]>0 and effects['local:fill']['ci991667'][0]>=-.005),
        heuristic_pass=bool(effects['heuristic:reward_day']['ci991667'][0]>0 and effects['heuristic:fill']['ci991667'][0]>=-.005),
        means={l:{m:float(np.mean([index[l,k][m] for k in keys])) for m in ['reward_day','fill']} for l in labels},
        price_counts={l:np.sum([index[l,k]['price_counts'] for k in keys],axis=0).tolist() for l in labels})


def collect(env,covers,key,label,models,counters,deadline):
    if label not in ['cross','local','baseline','heuristic']:raise ValueError('unknown controller')
    reset(env,key);trace=[]
    for day in range(1,29):
        if time.perf_counter()>deadline:raise TimeoutError('assessment time cap')
        pobs=env.pricing_observation().copy() if env.pricing_due else None;pa=None
        if pobs is not None:
            pa=(act(pobs,day,env.base_price,models[label]) if label in ['cross','local'] else
                threshold_act(pobs,[1.5]*10) if label=='heuristic' else np.full(10,4,dtype=int))
            env.select_price(pa)
        obs=env.replenishment_observation().copy();oa=order_action(obs,day,covers)
        _,reward,done,info=env.step(JointAction(None,oa));assert done==(day==28)
        counters['days']+=1
        trace.append(serial(dict(day=day,start=int(env.sim.start_day_index),obs=obs,actions=oa,
            pricing_obs=pobs,price_actions=pa,reward=reward,info=info,
            demand_seed=int(env.sim.scenario_seed),supplier_seed=int(env.pipeline.scenario_seed))))
    demand=sum(sum(r['info']['demand']) for r in trace);sales=sum(sum(r['info']['sales']) for r in trace)
    row=dict(key=key,label=label,reward_day=sum(r['reward'] for r in trace)/28,fill=sales/demand if demand else 1.,
        price_counts=np.bincount(np.concatenate([r['price_actions'] for r in trace if r['price_actions'] is not None]),minlength=5).tolist())
    return row,trace


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    prior=Path('outputs/structured_mappo/pricing_magnitude_v1/verification.json')
    assert json.loads(prior.read_text())['status']=='passed'
    paths={str(p) for folder in ['src','scripts'] for p in Path(folder).rglob('*.py')}
    paths|={str(p) for p in Path('data').rglob('*') if p.is_file() and ('aligned' in str(p) or 'sku' in p.name.lower())}
    paths|={'src/config/skus.json',INPUTS['calibration'],str(STUDY/'sku_seed211_final.json'),
        'CROSS_SKU_PRICING_PROTOCOL.md','CROSS_SKU_FEATURE_SPEC.md',str(prior)}
    hashes={p:digest(p) for p in sorted(paths)}
    covers=np.array(json.loads((STUDY/'sku_seed211_final.json').read_text())['mean'])
    write_json(out/'manifest.json',dict(hashes=hashes,covers=covers,maximum_days=501872,fit_cap=7200,assessment_cap=600,total_cap=7800))
    torch.set_num_threads(1);began=time.perf_counter();deadline=began+7200;counters=dict(days=0);stage='scan'
    try:
        env=make_research_env('S2',split='train',start=15,pricing_scope='per_sku',params=load_identifiable_training_calibration(INPUTS['calibration']))
        selections=[]
        with (out/'scan.jsonl').open('w') as handle:
            for ep,key in enumerate(range(2540000,2540128)):
                reset(env,key);trace=suffix(env,covers,0,4,counters,deadline)
                observations={r['day']:r['pricing_obs'] for r in trace if r['pricing_obs'] is not None}
                selected=[list(v) for v in select_candidates(observations,ep,env.base_price)]
                selections.append(dict(episode=ep,key=key,observations=observations,selected=selected))
                handle.write(json.dumps(dict(episode=ep,key=key,trace=trace),allow_nan=False)+'\n');handle.flush()
        selection=freeze_selection(out,selections,hashes,counters,env.base_price);selection_hash=digest(out/'selection_freeze.json')
        stage='branches';samples=[]
        with (out/'scan.jsonl').open() as scans,(out/'states.jsonl').open('w') as states,(out/'original_replays.jsonl').open('w') as originals:
            for line in scans:
                original=json.loads(line);ep=original['episode'];key=original['key'];chosen=selections[ep]['selected']
                reset(env,key);trace=[]
                while not env._finished:
                    day=int(env.day)
                    for rank,(selected_day,sku) in enumerate(chosen):
                        if day!=selected_day:continue
                        before=snapshot(env);obs=env.pricing_observation().tolist()
                        assert obs==selections[ep]['observations'][day]
                        probe=suffix(clone_environment(env),covers,sku,4,counters,deadline)
                        assert probe==original['trace'][day-1:];draws=[];replays=[];gains=[]
                        for rep in range(32):
                            keys=[int(np.random.SeedSequence([2550000,ep,day,sku,rep,z]).generate_state(1,dtype=np.uint64)[0]) for z in [0,1]]
                            branches=[]
                            for action in [0,4]:
                                branch=resample_future(env,keys);initial=snapshot(branch)
                                suffix_trace=suffix(branch,covers,sku,action,counters,deadline)
                                value=sum(.99**i*r['reward'] for i,r in enumerate(suffix_trace))
                                branches.append(dict(action=action,initial=initial,trace=suffix_trace,value=value))
                                if rep==0:
                                    replay=suffix(resample_future(env,keys),covers,sku,action,counters,deadline)
                                    assert replay==suffix_trace;replays.append(replay)
                            gains.append(branches[0]['value']-branches[1]['value']);draws.append(dict(keys=keys,branches=branches))
                        assert snapshot(env)==before
                        sample=dict(episode=ep,rank=rank,day=day,sku=sku,obs=obs,
                            features={arm:features(obs,day,env.base_price,arm)[sku].tolist() for arm in ['local','cross']},
                            target=float(np.mean(gains)/env.base_price[sku]),gains=gains)
                        samples.append(sample)
                        states.write(json.dumps(dict(sample=sample,before=before,after=snapshot(env),probe=probe,draws=draws,replays=replays),allow_nan=False)+'\n');states.flush()
                    r,_=advance(env,covers,day,0,4,counters,deadline);trace.append(r)
                assert trace==original['trace']
                originals.write(json.dumps(dict(episode=ep,key=key,trace=trace),allow_nan=False)+'\n');originals.flush()
                write_json(out/'progress.json',dict(stage=stage,windows=ep+1,days=counters['days'],seconds=time.perf_counter()-began))
                print(f'Cross-SKU pricing: {ep+1}/128 fit windows',flush=True)
        # Collection follows chronological days; fit in frozen selection-rank order.
        samples.sort(key=lambda r:(r['episode'],r['rank']))
        assert [(r['episode'],r['rank']) for r in samples]==[(ep,j) for ep in range(128) for j in [0,1]]
        assert counters['days']==7168+67*selection['suffix_length_sum']
        matrices={arm:np.array([r['features'][arm] for r in samples]) for arm in ['local','cross']}
        targets=np.array([r['target'] for r in samples]);models={arm:fit(matrices[arm],targets,arm) for arm in matrices}
        write_json(out/'policy.json',dict(models=models,matrices=matrices,targets=targets,samples=samples,base_prices=env.base_price))
        fit_seconds=time.perf_counter()-began;assert fit_seconds<=7200
        assert digest(out/'selection_freeze.json')==selection_hash and all(digest(p)==h for p,h in hashes.items())
        fit_hashes={p:digest(out/p) for p in ['scan.jsonl','selection_freeze.json','states.jsonl','original_replays.jsonl','policy.json']}
        write_json(out/'assessment_freeze.json',dict(fit_hashes=fit_hashes,source_hashes=hashes,days=counters['days'],fit_seconds=fit_seconds))
        freeze_hash=digest(out/'assessment_freeze.json');stage='assessment';assessment_start=time.perf_counter();rows=[];first={}
        with (out/'assessment.jsonl').open('w') as handle:
            for key in range(2560000,2560128):
                for label in ['cross','local','baseline','heuristic']:
                    row,trace=collect(env,covers,key,label,models,counters,min(began+7800,assessment_start+600));rows.append(row)
                    handle.write(json.dumps(dict(label=label,key=key,trace=trace,result=row),allow_nan=False)+'\n')
                    if key==2560000:first[label]=trace
                handle.flush()
        with (out/'assessment_replays.jsonl').open('w') as handle:
            for label in ['cross','local','baseline','heuristic']:
                _,trace=collect(env,covers,2560000,label,models,counters,min(began+7800,assessment_start+600))
                assert trace==first[label];handle.write(json.dumps(dict(label=label,key=2560000,trace=trace),allow_nan=False)+'\n')
        assessment_seconds=time.perf_counter()-assessment_start;elapsed=time.perf_counter()-began
        assert counters['days']==selection['planned_days']<=501872 and assessment_seconds<=600 and elapsed<=7800
        assert all(digest(p)==h for p,h in hashes.items()) and all(digest(out/p)==h for p,h in fit_hashes.items())
        assert digest(out/'assessment_freeze.json')==freeze_hash
        analysis=assess(rows)
        write_json(out/'summary.json',dict(status='complete',analysis=analysis,rows=rows,days=counters['days'],
            suffix_length_sum=selection['suffix_length_sum'],fit_seconds=fit_seconds,assessment_seconds=assessment_seconds,
            seconds=elapsed,test_scoring=False,freeze_hash=freeze_hash,selection_hash=selection_hash,
            trace_hashes={p.name:digest(p) for p in out.glob('*.jsonl')}))
        print(json.dumps(analysis),flush=True)
    except Exception as exc:
        write_json(out/'failed_status.json',dict(stage=stage,error=str(exc),counters=counters));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)

