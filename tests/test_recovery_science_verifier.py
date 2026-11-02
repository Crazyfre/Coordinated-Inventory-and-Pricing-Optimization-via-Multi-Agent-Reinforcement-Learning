import copy
import numpy as np
import pytest
from scripts import verify_recovery_science as verifier
from scripts import recovery_campaign_science as producer
from scripts.recovery_campaign_plan import build,identity
from tests.test_recovery_campaign_plan import study


def test_independent_complete_plan_rejects_rehashed_changes():
    config=study()
    plan=build(config)
    verifier.check_plan(config,plan)
    for field,value in [('days',0),('phase','assessment'),('seconds',1.)]:
        bad=copy.deepcopy(plan)
        bad['units'][129][field]=value
        with pytest.raises(AssertionError):verifier.check_plan(config,bad)
    bad=copy.deepcopy(plan)
    bad['units'][130]['specification']['dependencies']=[]
    bad['units'][130]['input_identity']=identity(dict(study=bad['identity'],specification=bad['units'][130]['specification']))
    with pytest.raises(AssertionError):verifier.check_plan(config,bad)


def test_independent_selection_and_normalized_fits():
    rng=np.random.default_rng(333)
    base=np.arange(1,11,dtype=float)
    samples=[]
    for ep in range(128):
        obs={}
        for day in (1,8,15,22):
            x=rng.uniform(0,30,204)
            x[150:160]=base*1.2
            obs[day]=x.tolist()
        chosen=verifier.selection(obs,ep,base,910100000)
        assert chosen==producer.select(obs,ep,base,910100000)
        for rank,(day,sku) in enumerate(chosen):
            samples.append(dict(episode=ep,rank=rank,day=day,sku=sku,obs=obs[day],gains=rng.normal(size=32).tolist()))
    policy=producer.fit_samples(samples,base)
    verifier.check_models(samples,base,policy)
    policy['models']['cross']['weights'][0]+=.01
    with pytest.raises(AssertionError):verifier.check_models(samples,base,policy)


def test_independent_intervals_and_changed_gate_rejection():
    ns=dict(assessment=910300000,bootstrap=910400000)
    units=[dict(episode=e,records=[dict(result=dict(label=l,key=ns['assessment']+e,
        reward_day=float(e%7)+(1. if l=='cross' else 0.),fill=.7,price_counts=[0,0,0,0,40]))
        for l in ('cross','local','baseline','heuristic')]) for e in range(128)]
    summary=producer.analyze(units,ns)
    rows=[r['result'] for u in units for r in u['records']]
    verifier.check_analysis(rows,summary,ns)
    summary['primary_pass']=False
    with pytest.raises(AssertionError):verifier.check_analysis(rows,summary,ns)


def test_paired_state_trace_keys_return_and_prefix_audit():
    ns=dict(scan=910100000,future=910200000)
    key=ns['scan'];day=22;sku=3
    zero=[0]*10;obs=[0]*260;pobs=[0]*204
    pobs[150:160]=[1.2]*10;pobs[-20:-10]=[1]*10
    start=int(np.random.default_rng(np.random.SeedSequence([key,314159])).integers(15,1859))
    def trace(first,keys,action=4):
        rows=[]
        for d in range(first,29):
            pa=None
            if (d-1)%7==0:
                pa=[4]*10
                if d==first:pa[sku]=action
                prices=np.array([.8,.9,1.,1.1,1.2])[pa].tolist()
            rows.append(dict(day=d,start=start,demand_seed=keys[0],supplier_seed=keys[1],obs=obs,
                actions=zero,pricing_obs=pobs if pa is not None else None,price_actions=pa,reward=0.,
                info=dict(price=prices,total_on_hand=0.,orders_delivered=zero,orders_clipped=zero,
                          reward_terms=dict(revenue=0.,ordering_cost=0.,holding_cost=0.,stockout_cost=0.,price_change_cost=0.))))
        return rows
    original=trace(1,verifier.original_noise_keys(key))
    before=dict(day=day,pending=None,start=start,inventory=zero,history=[[0]*10 for _ in range(14)],
                obs=obs,accepted=zero,clipped=zero)
    groups=[]
    for g in range(8):
        draws=[]
        for rep in range(g*4,g*4+4):
            keys=producer.keys(ns['future'],0,day,sku,rep)
            draws.append(dict(replica=rep,keys=keys,branches=[dict(action=a,initial=before,
                trace=trace(day,keys,a),value=0.) for a in (0,4)]))
        groups.append(dict(episode=0,rank=0,group=g,key=key,day=day,sku=sku,
                           before=before,prefix=original[:day-1],draws=draws))
    first=groups[0]['draws'][0]
    state=dict(rank=0,day=day,sku=sku,before=before,obs=pobs,probe=original[day-1:],
               keys=first['keys'],replays=[b['trace'] for b in first['branches']])
    sample,count=verifier.check_state_groups(original,state,groups,0,0,ns,np.zeros(10),np.ones(10))
    assert sample['gains']==[0.]*32 and count==8*21+67*7
    for field in ('key','return','prefix'):
        bad=copy.deepcopy(groups)
        if field=='key':bad[1]['draws'][0]['keys'][0]+=1
        elif field=='return':bad[1]['draws'][0]['branches'][0]['value']=1.
        else:bad[1]['prefix'][0]['reward']=1.
        with pytest.raises(AssertionError):
            verifier.check_state_groups(original,state,bad,0,0,ns,np.zeros(10),np.ones(10))
