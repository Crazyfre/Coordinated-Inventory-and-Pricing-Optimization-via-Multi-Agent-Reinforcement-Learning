"""Scientific work units for the fresh segmented protocol; no launch entry point."""
import numpy as np
from src.algos.cross_sku_pricing_features import features as pressure
from src.algos.cross_sku_rollout import matrix,fit
from scripts.screen_cross_sku_pricing import reset,collect
from scripts.diagnose_conditional_pricing import advance,suffix
from scripts.diagnose_counterfactual_rollouts import clone_environment
from scripts.audit_repeated_futures import snapshot,resample_future

LABELS=['cross','local','baseline','heuristic']


def plan():
    return ([f'scan_{e:03d}' for e in range(128)]+['selection']+
        [f'branch_{e:03d}' for e in range(128)]+['fit']+
        [f'assess_{e:03d}' for e in range(128)]+['replays','summary'])


def select(observations,ep,base):
    if set(observations)!={1,8,15,22}:raise ValueError('all weekly observations required')
    candidates=[(d,i) for d in [1,8,15,22] for i in range(10)]
    scores=np.concatenate([pressure(observations[d],base)[:,3] for d in [1,8,15,22]])
    best=int(np.argmax(scores));remaining=[i for i in range(40) if i!=best]
    rng=np.random.default_rng(np.random.SeedSequence([2570000,ep,991]))
    return [list(candidates[best]),list(candidates[remaining[int(rng.integers(39))]])]


def scan(env,covers,ep,deadline):
    reset(env,2570000+ep);counter=dict(days=0)
    trace=suffix(env,covers,0,4,counter,deadline)
    obs={r['day']:r['pricing_obs'] for r in trace if r['pricing_obs'] is not None}
    assert counter['days']==28
    return dict(episode=ep,key=2570000+ep,trace=trace,observations=obs,
                selected=select(obs,ep,env.base_price)),28


def freeze(scans,base,scan_hashes,source_hashes):
    if len(scans)!=128 or [r['episode'] for r in scans]!=list(range(128)):
        raise ValueError('all128 ordered scans required')
    if len(scan_hashes)!=128:raise ValueError('all128 scan artifact hashes required')
    selections=[];length=0
    for r in scans:
        ep=r['episode'];obs={int(k):v for k,v in r['observations'].items()}
        if r['key']!=2570000+ep or len(r['trace'])!=28:raise ValueError('fresh key and full original required')
        actual={v['day']:v['pricing_obs'] for v in r['trace'] if v['pricing_obs'] is not None}
        if obs!=actual or r['selected']!=select(obs,ep,base):raise ValueError('selection/trace mismatch')
        length+=sum(29-d for d,i in r['selected'])
        selections.append(dict(episode=ep,key=r['key'],observations=obs,selected=r['selected']))
    return dict(selections=selections,suffix_length_sum=length,planned_days=21616+67*length,
                scan_hashes=scan_hashes,source_hashes=source_hashes),0


def branch(env,covers,original,selection,deadline):
    ep=original['episode'];key=2570000+ep;counter=dict(days=0);trace=[];states=[]
    assert original['key']==selection['key']==key and selection['episode']==ep
    chosen=selection['selected'];reset(env,key)
    while not env._finished:
        day=int(env.day)
        for rank,(selected_day,sku) in enumerate(chosen):
            if day!=selected_day:continue
            before=snapshot(env);obs=env.pricing_observation().tolist()
            assert obs==selection['observations'].get(day,selection['observations'].get(str(day)))
            probe=suffix(clone_environment(env),covers,sku,4,counter,deadline)
            assert probe==original['trace'][day-1:];draws=[];replays=[];gains=[]
            for rep in range(32):
                keys=[int(np.random.SeedSequence([2580000,ep,day,sku,rep,z]).generate_state(1,dtype=np.uint64)[0]) for z in [0,1]]
                branches=[]
                for action in [0,4]:
                    future=resample_future(env,keys);initial=snapshot(future)
                    part=suffix(future,covers,sku,action,counter,deadline)
                    value=sum(.99**i*r['reward'] for i,r in enumerate(part))
                    branches.append(dict(action=action,initial=initial,trace=part,value=value))
                    if rep==0:
                        replay=suffix(resample_future(env,keys),covers,sku,action,counter,deadline)
                        assert replay==part;replays.append(replay)
                gains.append(branches[0]['value']-branches[1]['value']);draws.append(dict(keys=keys,branches=branches))
            after=snapshot(env);assert after==before
            sample=dict(episode=ep,rank=rank,day=day,sku=sku,obs=obs,gains=gains,
                target=float(np.mean(gains)/env.base_price[sku]),
                features={arm:matrix(obs,day,env.base_price,arm)[sku].tolist() for arm in ['local','cross']})
            states.append(dict(sample=sample,before=before,after=after,probe=probe,draws=draws,replays=replays))
        step,_=advance(env,covers,day,0,4,counter,deadline);trace.append(step)
    assert trace==original['trace'] and len(states)==2
    assert counter['days']==28+67*sum(29-d for d,i in chosen)
    return dict(episode=ep,key=key,original_replay=trace,states=states),counter['days']


def fit_models(branches,base,selection_hash,branch_hashes,source_hashes):
    if len(branches)!=128 or [r['episode'] for r in branches]!=list(range(128)):
        raise ValueError('all128 ordered branch units required')
    if len(branch_hashes)!=128:raise ValueError('complete branch hashes required')
    samples=sorted([s['sample'] for r in branches for s in r['states']],key=lambda s:(s['episode'],s['rank']))
    if [(s['episode'],s['rank']) for s in samples]!=[(e,r) for e in range(128) for r in [0,1]]:
        raise ValueError('complete unique256 samples required')
    matrices={arm:np.array([s['features'][arm] for s in samples]) for arm in ['local','cross']}
    targets=np.array([s['target'] for s in samples]);models={arm:fit(matrices[arm],targets,arm) for arm in matrices}
    return dict(samples=samples,base_prices=np.asarray(base).tolist(),matrices={k:v.tolist() for k,v in matrices.items()},
        targets=targets.tolist(),models=models,selection_hash=selection_hash,branch_hashes=branch_hashes,source_hashes=source_hashes),0


def assessment(env,covers,ep,models,deadline):
    counter=dict(days=0);records=[]
    for label in LABELS:
        row,trace=collect(env,covers,2590000+ep,label,models,counter,deadline)
        records.append(dict(label=label,key=2590000+ep,result=row,trace=trace))
    assert counter['days']==112
    return dict(episode=ep,records=records),112


def replays(env,covers,models,first,deadline):
    result,days=assessment(env,covers,0,models,deadline)
    assert result==first
    return result,days


def analyze(units):
    if len(units)!=128 or [u['episode'] for u in units]!=list(range(128)):raise ValueError('all128 ordered assessment units required')
    rows=[r['result'] for u in units for r in u['records']];ix={(r['label'],r['key']):r for r in rows}
    keys=list(range(2590000,2590128))
    if len(ix)!=len(rows) or set(ix)!={(l,k) for l in LABELS for k in keys}:raise ValueError('complete unique assessment grid required')
    draws=np.random.default_rng(20261108).integers(128,size=(10000,128));effects={}
    for ref in ['baseline','local','heuristic']:
        for metric in ['reward_day','fill']:
            d=np.array([ix['cross',k][metric]-ix[ref,k][metric] for k in keys])
            effects[f'{ref}:{metric}']=dict(mean=float(d.mean()),ci991667=np.quantile(d[draws].mean(1),[.05/12,1-.05/12]).tolist())
    passes={r:bool(effects[f'{r}:reward_day']['ci991667'][0]>0 and effects[f'{r}:fill']['ci991667'][0]>=-.005) for r in ['baseline','local','heuristic']}
    return dict(effects=effects,primary_pass=bool(passes['baseline'] and effects['baseline:reward_day']['mean']>=.5),
        representation_pass=passes['local'],heuristic_pass=passes['heuristic'],
        means={l:{m:float(np.mean([ix[l,k][m] for k in keys])) for m in ['reward_day','fill']} for l in LABELS},
        price_counts={l:np.sum([ix[l,k]['price_counts'] for k in keys],axis=0).tolist() for l in LABELS})
