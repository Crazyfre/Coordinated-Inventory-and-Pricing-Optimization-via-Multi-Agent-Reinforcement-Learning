"""Independent CENTRAL/LOCAL learned support replay; no collect/update/refit calls."""
import numpy as np
import torch
from outputs.conditional_credit_pilot_replay import choose,serial,trace_digest
from src.algos.parameter_policy_gradient import order_action
from src.algos.structured_pricing_ppo import critic_features
from src.env.interface import JointAction
from src.experiments.pipeline_headroom import clone_env,fingerprint


@torch.no_grad()
def critic_value(model,state,role):
    network=model.critics[role if model.mode=='local' else 'shared']
    value=network(torch.as_tensor(state)).squeeze();norm=model.value_norm
    if not int(norm.updates):return float(value)
    mean=(norm.mean_ema/norm.mass).float()
    std=(norm.square_ema/norm.mass-(norm.mean_ema/norm.mass).square()).clamp_min(.01).sqrt().float()
    return float(value*std+mean)

def replay_integrated_control(model,env,item,advance,identity_seen,*,clone=clone_env,stamp=fingerprint):
    assert torch.equal(torch.get_rng_state(),item['policy_rng_before'])
    key=item['key'];env.reset(seed=key);coordinate_rng=np.random.default_rng(np.random.SeedSequence([key,6179]))
    iterators={r:iter(item['rows'][r]) for r in ('pricing','ordering')};roots=[];trace=[];rewards=[];leads=[]
    def root(role,day):
        i=int(coordinate_rng.integers(10));obs,f,action,logps,_=choose(model,env,role,day);row=next(iterators[role])
        assert row['start']==day-1 and row['coordinate']==i and np.array_equal(f,row['features']) and np.array_equal(action,row['action'])
        assert np.array_equal(logps,row['component_logps']) and row['logp']==float(logps[i]) and row['joint_logp']==float(logps.sum())
        state=(np.r_[critic_features(obs,day),np.eye(10)[i]] if model.mode=='local' else
               np.r_[critic_features(env.get_state(),day),[1.,0.] if role=='pricing' else [0.,1.],np.eye(10)[i]]).astype(np.float32)
        assert np.array_equal(state,row['state']) and row['value']==critic_value(model,state,role)
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
        i=row['coordinate'];first=row['start']+1;entries=saved['entries'];support=list(range(5)) if role=='pricing' else np.unique(np.clip(int(row['features'][i,-1])+np.array([-1,0,1]),0,40)).tolist()
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
