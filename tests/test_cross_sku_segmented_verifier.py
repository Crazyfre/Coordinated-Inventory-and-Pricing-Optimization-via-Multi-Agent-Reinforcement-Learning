import copy
import numpy as np
import pytest
from scripts import cross_sku_segmented_units as units
from scripts.verify_cross_sku_segmented import selection, feature_matrix, check_ridge, check_analysis, audit_trace
from src.algos.cross_sku_rollout import matrix, fit


def test_fresh_selection_features_and_independent_normalized_fits():
    rng=np.random.default_rng(213);base=rng.uniform(1,6,10)
    obs={d:rng.uniform(0,90,204) for d in [1,8,15,22]}
    for d,x in obs.items():
        x[-20:-10]=rng.uniform(1,15,10)
        for arm in ['local','cross']:
            np.testing.assert_allclose(feature_matrix(x,d,base,arm),matrix(x,d,base,arm),rtol=0,atol=1e-14)
    for ep in range(128):assert selection(obs,ep,base)==units.select(obs,ep,base)
    tied={d:np.zeros(204) for d in obs}
    assert selection(tied,0,base)[0]==[1,0]
    for arm,width in [('local',16),('cross',20)]:
        x=rng.normal(size=(256,width));x[:,13]=4.;y=rng.normal(size=256)
        model=fit(x,y,arm);check_ridge(x,y,model)
        for field in ['weights','mean','scale']:
            bad=copy.deepcopy(model);bad[field][11]+=.25
            with pytest.raises(AssertionError):check_ridge(x,y,bad)


def test_variable_effect_intervals_service_and_separate_gates():
    rng=np.random.default_rng(214);data=[]
    for ep in range(128):
        records=[]
        for label in units.LABELS:
            records.append(dict(result=dict(label=label,key=2590000+ep,
                reward_day=float(90+rng.normal()+(2 if label=='cross' else 0)),
                fill=.69 if label=='cross' else .7,price_counts=[0,0,0,0,40])))
        data.append(dict(episode=ep,records=records))
    a=units.analyze(data);rows=[r['result'] for u in data for r in u['records']]
    check_analysis(rows,a)
    assert not a['primary_pass'] and not a['representation_pass'] and not a['heuristic_pass']
    for ref in ['baseline','local','heuristic']:
        for metric in ['reward_day','fill']:
            bad=copy.deepcopy(a);bad['effects'][f'{ref}:{metric}']['ci991667'][0]+=.1
            with pytest.raises(AssertionError):check_analysis(rows,bad)
    with pytest.raises(AssertionError):check_analysis(rows[:-1],a)


def synthetic_trace(first=1,sku=-1,action=4,label='baseline'):
    base=np.ones(10);covers=np.ones(10);trace=[]
    for day in range(first,29):
        obs=np.zeros(260,np.float32);obs[-10:]=2
        pobs=np.zeros(204,np.float32);pobs[-20:-10]=2;pobs[150:160]=1
        pa=None
        if (day-1)%7==0:
            pa=[4]*10
            if sku>=0 and day==first:pa[sku]=action
            if label=='cross':pa[0]=0
            prices=(base*np.array([.8,.9,1.,1.1,1.2])[pa]).tolist()
        trace.append(dict(day=day,start=30,demand_seed=11,supplier_seed=12,obs=obs.tolist(),
            actions=[1]*10,pricing_obs=pobs.tolist() if pa else None,price_actions=pa,
            reward=6.,info=dict(price=prices,total_on_hand=0.,reward_terms=dict(
                revenue=10.,ordering_cost=1.,holding_cost=1.,stockout_cost=1.,price_change_cost=1.))))
    models={arm:dict(mean=[0.]*w,scale=[1.]*w,weights=[1.]*10+[0.]*(w-10)) for arm,w in [('local',16),('cross',20)]}
    return trace,covers,base,models


@pytest.mark.parametrize('mode',['baseline','cross','suffix'])
def test_trace_actions_economics_and_branch_price_hold(mode):
    first=22 if mode=='suffix' else 1;sku=3 if mode=='suffix' else -1
    label='cross' if mode=='cross' else 'baseline';action=0 if mode=='suffix' else 4
    trace,covers,base,models=synthetic_trace(first,sku,action,label)
    args=(first,30,[11,12],covers,base,models,sku,action,label)
    assert audit_trace(trace,*args)==sum(.99**i*6 for i in range(len(trace)))
    for field in ['actions','price_actions','reward','supplier_seed']:
        bad=copy.deepcopy(trace)
        if field in ['actions','price_actions']:bad[0][field][0]+=1
        else:bad[0][field]+=1
        with pytest.raises(AssertionError):audit_trace(bad,*args)
    bad=copy.deepcopy(trace);bad[1]['info']['price'][0]+=.1
    with pytest.raises(AssertionError):audit_trace(bad,*args)
    with pytest.raises(AssertionError):audit_trace(trace[:-1],*args)
