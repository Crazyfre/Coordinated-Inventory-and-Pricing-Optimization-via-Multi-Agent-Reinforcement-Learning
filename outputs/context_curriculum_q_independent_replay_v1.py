"""Independent economic-feature/Q-scale/coefficient and full branch replay.

Derived from the immutable full-price original replay loop; no producer update,
collection, feature or Q-table method is used. No execution on import.
"""
import numpy as np
import torch
from scipy.stats import nbinom
from outputs.conditional_credit_pilot_replay import serial,trace_digest
from src.algos.parameter_policy_gradient import order_action
from src.algos.structured_pricing_ppo import critic_features
from src.env.interface import JointAction
from src.experiments.pipeline_headroom import clone_env,fingerprint


def independent_price_features(observation,day,base_prices,phi):
    # Independent reconstruction; no producer feature/proxy/bin function.
    original=np.asarray(observation,dtype=np.float32)
    x=original.astype(float);base=np.asarray(base_prices,float);shape=np.asarray(phi,float)
    horizon=min(7,29-day);remaining=29-day
    prices=base[:,None]*np.array([.8,.9,1.,1.1,1.2])
    mean=horizon*x[-20:-10,None]*(prices/x[150:160,None])**-1.5
    stock,average,size=np.broadcast_arrays(x[:10,None],mean,horizon*shape[:,None])
    probability=size/(size+average);whole=np.floor(stock)
    sold=average*nbinom.cdf(whole-1,size+1,probability)+stock*nbinom.sf(whole,size,probability)
    exposure=.6*base*max(0.,1.-4.5/remaining)
    score=((prices-exposure[:,None])*sold-(mean-sold)
        -.02*horizon*(x[:10,None]-sold/2)-.05*np.abs(prices-x[150:160,None]))
    cover=np.clip(original[:10]/(1+7*original[-20:-10]),0,3)
    result=[]
    for i in range(10):
        cb=sum(float(cover[i])>=edge for edge in (.25,.5,1.,2.))
        result.append([(((day-1)//7)*5+cb)*9+sum(float(score[i,a]-score[i,4])>=edge
            for edge in (-128.,-16.,-2.,-.25,0.,.25,2.,16.)) for a in range(4)])
    return np.array(result,dtype=np.float32)


def independent_price_distribution(actor,features):
    index=torch.tensor(features,dtype=torch.long)
    columns=torch.arange(4)
    residual=(8.*actor.weight[index,columns]).clamp(-6,6)
    logits=torch.cat((residual,torch.zeros(residual.shape[:-1]+(1,))),dim=-1)
    logits=logits+torch.log(torch.tensor([.1875,.1875,.1875,.1875,.25]))
    return torch.distributions.Categorical(logits=logits/actor.temperature)


@torch.no_grad()
def independently_raw_and_beta(model,state,role,features):
    expanded=np.repeat(np.asarray(state,dtype=np.float32)[None,:],10,axis=0)
    expanded[:,-10:]=np.eye(10);inputs=torch.tensor(expanded)
    if model.mode=='central':
        network=model.critics['shared'];values=network.heads[role](network.trunk(inputs))
    else:
        values=model.critics['local'].networks[role](inputs)
    index=0 if role=='pricing' else 1
    anchors=torch.full((10,),4,dtype=torch.long) if role=='pricing' else torch.tensor(features[:,-1],dtype=torch.long)
    difference=values-values[torch.arange(10),anchors][:,None]
    raw=(difference*model.role_scales.scales[index].float()).numpy()
    moments=model.q_weight
    beta=(moments.qy[index]/moments.qq[index].clamp_min(1e-12)).clamp(0,1).float() if moments.qq[index]>1e-12 else torch.tensor(0.)
    return raw,float(beta)


@torch.no_grad()
def critic_value(model,state,role):return 0.


@torch.no_grad()
def choose(model,env,role,day,greedy=False):
    observation=env.pricing_observation().copy() if role=='pricing' else env.replenishment_observation().copy()
    if role=='pricing':
        assert np.array_equal(env.base_price,model.pricing.base_prices.numpy())
        f=independent_price_features(observation,day,model.pricing.base_prices.numpy(),model.pricing.phi.numpy())
        d=independent_price_distribution(model.pricing,f)
        assert torch.equal(d.probs,model.pricing.distribution(torch.tensor(f)).probs)
    else:
        from src.algos.pipeline_structured_mappo import pipeline_features
        f=np.c_[pipeline_features(observation,day,'timing'),order_action(observation,day,model.ordering.covers.numpy())].astype(np.float32)
        d=model.ordering.distribution(torch.tensor(f))
    action=d.probs.argmax(-1) if greedy else d.sample()
    lp=d.log_prob(action)
    return observation,f,action.numpy(),lp.numpy(),float(lp.sum())


def independent_price_support(model,features,i):
    return list(range(5))

def replay_role_calibrated_q_control(model,env,item,advance,identity_seen,*,clone=clone_env,stamp=fingerprint):
    assert torch.equal(torch.get_rng_state(),item['policy_rng_before'])
    key=item['key'];env.reset(seed=key);coordinate_rng=np.random.default_rng(np.random.SeedSequence([key,6179]))
    iterators={r:iter(item['rows'][r]) for r in ('pricing','ordering')};roots=[];trace=[];rewards=[];leads=[]
    def root(role,day):
        obs,f,action,logps,_=choose(model,env,role,day);row=next(iterators[role])
        if role=='pricing':
            multiplier=10;i=int(coordinate_rng.integers(10))
        else:i=int(coordinate_rng.integers(10));multiplier=10
        assert row['coordinate_multiplier']==multiplier
        assert row['start']==day-1 and row['coordinate']==i and np.array_equal(f,row['features']) and np.array_equal(action,row['action'])
        assert np.array_equal(logps,row['component_logps']) and row['logp']==float(logps[i]) and row['joint_logp']==float(logps.sum())
        state=(np.r_[critic_features(obs,day),np.eye(10)[i]] if model.mode=='local' else
               np.r_[critic_features(env.get_state(),day),[1.,0.] if role=='pricing' else [0.,1.],np.eye(10)[i]]).astype(np.float32)
        assert np.array_equal(state,row['state']) and row['value']==critic_value(model,state,role)
        raw,beta=independently_raw_and_beta(model,state,role,f)
        assert np.array_equal(row['q_pre'],raw*np.float32(beta))
        assert np.array_equal(row['q_raw_pre'],raw) and row['q_beta_pre']==beta
        reference=action.copy();reference[i]=4 if role=='pricing' else order_action(obs,day,model.ordering.covers.detach().numpy())[i]
        assert np.array_equal(reference,row['reference_action'])
        saved=item['roots'][len(roots)];rng=torch.get_rng_state().clone()
        assert saved['role']==role and torch.equal(saved['root_rng'],rng)
        roots.append((role,row,clone(env),rng,saved));return action
    for day in range(1,29):
        price=root('pricing',day) if env.pricing_due else None
        if price is not None:env.select_price(price)
        action=root('ordering',day);obs=env.replenishment_observation().copy()
        _,r,done,info=advance(env,JointAction(None,action),'actual');assert done==(day==28)
        rewards.append(float(r));leads.append(env._last_lead_times.copy())
        trace.append(dict(day=day,obs=obs,price_actions=price,actions=action,reward=float(r),info=info))
    terminal=torch.get_rng_state().clone();assert torch.equal(terminal,item['actual_policy_rng'])
    assert rewards==item['rewards'] and serial(trace)==serial(item['trace']) and len(roots)==len(item['roots'])==32
    for it in iterators.values():assert next(it,None) is None
    parent=stamp(env);base_plan=[];extra_plan=[];targets={r:[] for r in iterators}
    skips=selfchecks=common_pairs=same_price_pairs=sampled_skips=0;sampled_days=0
    for role,row,snapshot,rng,saved in roots:
        i=row['coordinate'];first=row['start']+1;entries=saved['entries'];support=independent_price_support(model,row['features'],i) if role=='pricing' else np.unique(np.clip(int(row['features'][i,-1])+np.array([-1,0,1]),0,40)).tolist()
        assert [e['action'] for e in entries]==support
        with torch.no_grad():masses=getattr(model,role).distribution(torch.tensor(row['features'])).probs[i].double().numpy()
        before=stamp(snapshot);expected_returns={};probabilities={}
        for entry in entries:
            value=entry['action'];probability=float(masses[value]);assert probability==entry['probability'] and probability>0
            probabilities[value]=probability;actual_identity=value==int(row['action'][i]);anchor=value==int(row['reference_action'][i])
            assert entry['source']==('actual' if actual_identity else 'strong' if anchor else 'branch')
            selfcheck=actual_identity and role not in identity_seen
            if actual_identity:
                skips+=1;identity_seen.add(role);rr=rewards[row['start']:];rt=trace[row['start']:]
            if not actual_identity or selfcheck:
                branch=clone(snapshot);torch.set_rng_state(rng);rr=[];rt=[]
                for day in range(first,29):
                    price=None
                    if branch.pricing_due:
                        if day==first and role=='ordering':
                            assert branch._pending_price is not None
                            price=np.abs((branch._pending_price/branch.base_price)[:,None]-np.asarray(branch.config.pricing.multipliers)).argmin(1)
                        else:
                            if day==first and role=='pricing':price=row['action'].copy();price[i]=value
                            else:price=choose(model,branch,'pricing',day)[2]
                            branch.select_price(price)
                    obs=branch.replenishment_observation().copy()
                    if day==first and role=='ordering':action=row['action'].copy();action[i]=value
                    else:action=choose(model,branch,'ordering',day)[2]
                    _,reward,done,info=advance(branch,JointAction(None,action),'identity_check' if selfcheck else 'branch');assert done==(day==28)
                    rr.append(float(reward));rt.append(dict(day=day,obs=obs,price_actions=price,actions=action,reward=float(reward),info=info))
                    if hasattr(env,'sim'):
                        mask=(np.asarray(action)>0)&(np.asarray(trace[day-1]['actions'])>0)
                        assert np.array_equal(branch._last_lead_times[mask],leads[day-1][mask]);common_pairs+=int(mask.sum())
                        if np.array_equal(info['price'],trace[day-1]['info']['price']):
                            assert np.array_equal(info['demand'],trace[day-1]['info']['demand']);same_price_pairs+=1
                if selfcheck:
                    assert rr==rewards[row['start']:] and serial(rt)==serial(trace[row['start']:])
                    assert torch.equal(torch.get_rng_state(),terminal);selfchecks+=1
                if not actual_identity:
                    part=dict(role=role,day=first,days=len(rr));(base_plan if anchor else extra_plan).append(part)
            assert rr==entry['rewards'] and trace_digest(rt)==entry['trace_sha256']
            value_return=float(np.dot(model.gamma**np.arange(len(rr)),rr)/100)
            assert abs(value_return-entry['discounted_return'])<1e-10;expected_returns[value]=value_return
            if anchor:assert rr==row['reference_rewards']
            assert stamp(snapshot)==before and stamp(env)==parent
        q0=expected_returns[int(row['reference_action'][i])]
        for entry in entries:assert abs(entry['residual_return']-(expected_returns[entry['action']]-q0))<1e-10
        target=sum(probabilities[a]*(q-q0) for a,q in expected_returns.items());targets[role].append(target)
        actual_return=sum(model.gamma**k*r for k,r in enumerate(rewards[row['start']:]))/100
        assert abs(row['target']-(actual_return-q0))<1e-10
        identity=np.array_equal(row['action'],row['reference_action'])
        assert row['branch_days']==(0 if identity else 29-first)
        sampled_skips+=int(identity);sampled_days+=row['branch_days']
    plan=base_plan+extra_plan;branch_days=sum(p['days'] for p in plan)
    assert item['reference_days']==branch_days and item['sampled_reference_days']==sampled_days and item['skipped_identical']==sampled_skips
    torch.set_rng_state(terminal);assert stamp(env)==parent
    return dict(targets=targets,branch_days=branch_days,skips=skips,selfchecks=selfchecks,common_pairs=common_pairs,
        same_price_pairs=same_price_pairs,branch_plan=plan)
