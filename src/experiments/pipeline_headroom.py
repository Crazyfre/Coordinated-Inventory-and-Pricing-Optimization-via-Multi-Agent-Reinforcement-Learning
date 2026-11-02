"""Local branch diagnostics and snapshot checks; no policy fitting."""
import copy
import hashlib
import json
import numpy as np
from src.algos.parameter_policy_gradient import order_action
from src.env.interface import JointAction


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value


def fingerprint(env):
    """Mutable environment state; functions excluded because step rebinds hook."""
    state=dict(day=env.day,finished=env._finished,inventory=env.inventory._on_hand,
        inventory_events=env.inventory.capacity_violation_events,inventory_clipped=env.inventory.capacity_clipped_units,
        pipeline=env.pipeline._in_flight,pipeline_seed=env.pipeline.scenario_seed,
        pipeline_rng=env.pipeline.rng.bit_generator.state,env_rng=env.np_random.bit_generator.state,
        demand_rng=env.sim.rng.bit_generator.state,demand_seed=env.sim.scenario_seed,
        start=env.sim.start_day_index,price=env._current_price,pending=env._pending_price,
        forecast_price=env.forecaster._price,demand_price=env.sim._price,last_mu=env.sim._last_mu,
        history=list(env._demand_history),leads=env._last_lead_times,
        arrivals=env._arrivals_today,accepted=env._accepted_today,clipped=env._clipped_today)
    return hashlib.sha256(json.dumps(serial(state),sort_keys=True,allow_nan=False).encode()).hexdigest()


def clone_env(env):
    before=fingerprint(env);clone=copy.deepcopy(env)
    assert clone is not env and clone.sim is not env.sim and clone.pipeline is not env.pipeline
    assert clone.inventory is not env.inventory and clone.forecaster is not env.forecaster
    assert clone.demand_source.sim is clone.sim and clone.forecaster.sim is clone.sim
    assert clone.np_random is not env.np_random and clone.sim.rng is not env.sim.rng
    assert not np.shares_memory(clone.inventory._on_hand,env.inventory._on_hand)
    for name in ('_current_price','_pending_price','_last_lead_times'):
        if getattr(env,name) is not None:assert not np.shares_memory(getattr(clone,name),getattr(env,name))
    for day,value in env.pipeline._in_flight.items():assert not np.shares_memory(clone.pipeline._in_flight[day],value)
    assert fingerprint(clone)==before==fingerprint(env)
    return clone


def candidates(anchor):
    q=np.asarray(anchor,dtype=np.int64)
    if q.shape!=(10,) or ((q<0)|(q>40)).any():raise ValueError('ten grid indices0..40 required')
    groups={tuple(q):dict(action=q.copy(),labels=['baseline'])}
    for sku in range(10):
        for shift in (-1,1):
            a=q.copy();a[sku]=np.clip(a[sku]+shift,0,40);identity=tuple(a)
            if identity not in groups:groups[identity]=dict(action=a,labels=[])
            groups[identity]['labels'].append(f'sku{sku}:{shift:+d}')
    assert sum(len(g['labels']) for g in groups.values())==21
    return list(groups.values())


def fixed_rules(obs,anchor):
    """Prespecified +one-grid move based on three-day forecast pressure."""
    x=np.asarray(obs,np.float32);q=np.asarray(anchor);pipe=x[10:80].reshape(10,7);forecast=x[-10:]
    result={}
    for name,incoming in (('timing_deficit',pipe[:,:2].sum(1)),('blind_deficit',pipe.sum(1))):
        risk=(3*forecast-x[:10]-incoming)/(1+forecast)
        feasible=np.where(q<40,risk,-np.inf);sku=int(np.argmax(feasible))
        result[name]=f'sku{sku}:+1' if feasible[sku]>0 else 'baseline'
    return result


def prepare(env,covers):
    price=np.full(10,4,dtype=np.int64) if env.pricing_due else None
    if price is not None:env.select_price(price)
    obs=env.replenishment_observation().copy()
    return obs,price,order_action(obs,env.day,covers)


def advance(env,obs,price,action):
    day=env.day;_,reward,done,info=env.step(JointAction(None,np.asarray(action,dtype=np.int64)))
    assert done==(day==28)
    return dict(day=day,obs=obs.copy(),price=price,action=np.asarray(action).copy(),
                reward=float(reward),info=copy.deepcopy(info),lead_times=env._last_lead_times.copy())


def totals(trace,gamma):
    return dict(discounted=float(sum(gamma**j*r['reward'] for j,r in enumerate(trace))),
                reward=float(sum(r['reward'] for r in trace)),
                demand=float(sum(np.asarray(r['info']['demand']).sum() for r in trace)),
                sales=float(sum(np.asarray(r['info']['sales']).sum() for r in trace)))


def assert_paired(branch,parent,*,baseline=False):
    assert len(branch)==len(parent);common=0
    for b,p in zip(branch,parent):
        assert b['day']==p['day']
        assert np.array_equal(b['info']['price'],p['info']['price'])
        assert np.array_equal(b['info']['demand'],p['info']['demand'])
        mask=(np.asarray(b['action'])>0)&(np.asarray(p['action'])>0)
        assert np.array_equal(np.asarray(b['lead_times'])[mask],np.asarray(p['lead_times'])[mask]);common+=int(mask.sum())
    if baseline:assert serial(branch)==serial(parent)
    return common
