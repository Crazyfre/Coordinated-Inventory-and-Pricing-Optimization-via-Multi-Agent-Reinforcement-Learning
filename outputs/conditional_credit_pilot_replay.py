"""Independent manual rollout/label replay. Does not call collect/continuation/update.

All nonidentity suffixes are simulated. Identity suffixes reuse the independently
replayed actual path, after reconstructing the root action, policy RNG and state.
One identity root per role/arm/batch is ALSO fully simulated, prospectively chosen.
"""
import copy
import hashlib
import json
import numpy as np
import torch
from src.algos.parameter_policy_gradient import order_action
from src.algos.pipeline_structured_mappo import pipeline_features
from src.algos.structured_pricing_ppo import pricing_features,critic_features
from src.env.interface import JointAction
from src.experiments.pipeline_headroom import clone_env,fingerprint


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value


def trace_digest(value):
    return hashlib.sha256(json.dumps(serial(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@torch.no_grad()
def choose(model,env,role,day,*,greedy=False):
    obs=env.pricing_observation().copy() if role=='pricing' else env.replenishment_observation().copy()
    if role=='pricing':f=pricing_features(obs,day,env.base_price)
    else:
        covers=model.ordering.covers.detach().numpy()
        f=np.c_[pipeline_features(obs,day,'timing'),order_action(obs,day,covers)].astype(np.float32)
    dist=getattr(model,role).distribution(torch.as_tensor(f));action=dist.probs.argmax(-1) if greedy else dist.sample()
    logp=dist.log_prob(action)
    return obs,f,action.numpy(),logp.numpy(),float(logp.sum())


@torch.no_grad()
def critic_value(model,state):
    raw=model.critics['shared'](torch.as_tensor(state)).squeeze()
    norm=model.value_norm
    if not int(norm.updates):return float(raw)
    mean=(norm.mean_ema/norm.mass).float()
    std=(norm.square_ema/norm.mass-(norm.mean_ema/norm.mass).square()).clamp_min(.01).sqrt().float()
    return float(raw*std+mean)


def replay_greedy(model,env,key,advance,kind):
    env.reset(seed=key);trace=[];rewards=[]
    for day in range(1,29):
        price=choose(model,env,'pricing',day,greedy=True)[2] if env.pricing_due else None
        if price is not None:env.select_price(price)
        obs,_,action,_,_=choose(model,env,'ordering',day,greedy=True)
        _,reward,done,info=advance(env,JointAction(None,action),kind);assert done==(day==28)
        rewards.append(float(reward));trace.append(dict(day=day,obs=obs,price_actions=price,actions=action,reward=float(reward),info=info))
    return dict(rewards=rewards,trace=trace)


def replay_training(model,arm,env,item,advance,identity_seen,*,clone=clone_env,stamp=fingerprint):
    """Caller restores exact PRE-BATCH checkpoint and preceding episode RNG."""
    key=item['key'];assert torch.equal(torch.get_rng_state(),item['policy_rng_before'])
    env.reset(seed=key);coordinates=np.random.default_rng(np.random.SeedSequence([key,6179]))
    selected={role:iter(item['rows'][role]) for role in ('pricing','ordering')}
    rewards=[];trace=[];roots=[];lead_rows=[]
    def root(role,day):
        index=int(coordinates.integers(10)) if arm!='standard' else None
        obs,f,action,lp,full=choose(model,env,role,day);row=next(selected[role])
        assert row['start']==day-1 and np.array_equal(row['features'],f) and np.array_equal(row['action'],action)
        state=np.r_[critic_features(env.get_state(),day),[1.,0.] if role=='pricing' else [0.,1.]]
        if arm!='standard':state=np.r_[state,np.eye(10)[index]]
        state=state.astype(np.float32);assert np.array_equal(row['state'],state)
        assert row['value']==critic_value(model,state)
        if arm=='standard':assert row['logp']==full
        else:
            assert row['coordinate']==index and np.array_equal(row['component_logps'],lp)
            assert row['logp']==float(lp[index]) and row['joint_logp']==float(lp.sum())
            expected=action.copy();expected[index]=4 if role=='pricing' else order_action(obs,day,model.ordering.covers.detach().numpy())[index]
            assert np.array_equal(row['reference_action'],expected)
            saved=item['roots'][len(roots)];rng=torch.get_rng_state().clone()
            assert saved['role']==role and torch.equal(rng,saved['root_rng'])
            roots.append((role,row,clone(env),rng,saved))
        return action
    for day in range(1,29):
        price=root('pricing',day) if env.pricing_due else None
        if price is not None:env.select_price(price)
        action=root('ordering',day);obs=env.replenishment_observation().copy()
        _,reward,done,info=advance(env,JointAction(None,action),'actual');assert done==(day==28)
        rewards.append(float(reward));lead_rows.append(env._last_lead_times.copy())
        trace.append(dict(day=day,obs=obs,price_actions=price,actions=action,reward=float(reward),info=info))
    terminal=torch.get_rng_state().clone();assert torch.equal(terminal,item['actual_policy_rng'])
    assert rewards==item['rewards'] and serial(trace)==serial(item['trace'])
    for rows in selected.values():assert next(rows,None) is None
    assert len(roots)==(0 if arm=='standard' else 32)==len(item['roots'])
    parent=stamp(env);branches=skips=selfchecks=common_pairs=same_price=0;plan=[]
    for role,row,snapshot,rng,saved in roots:
        identity=np.array_equal(row['action'],row['reference_action'])
        selfcheck=identity and role not in identity_seen
        if identity:
            skips+=1;identity_seen.add(role)
            # root snapshot and RNG were independently reconstructed above;
            # unchanged root + frozen policy + same future randomness = actual suffix.
            rr=rewards[row['start']:];rt=trace[row['start']:]
            assert row['branch_days']==0
        if not identity or selfcheck:
            before=stamp(snapshot);branch=clone(snapshot);torch.set_rng_state(rng);rr=[];rt=[];first=branch.day
            for day in range(first,29):
                price=None
                if branch.pricing_due:
                    if day==first and role=='ordering':
                        assert branch._pending_price is not None
                        price=np.abs((branch._pending_price/branch.base_price)[:,None]-np.asarray(branch.config.pricing.multipliers)).argmin(1)
                    else:
                        price=row['reference_action'].copy() if day==first and role=='pricing' else choose(model,branch,'pricing',day)[2]
                        branch.select_price(price)
                obs=branch.replenishment_observation().copy()
                action=row['reference_action'].copy() if day==first and role=='ordering' else choose(model,branch,'ordering',day)[2]
                _,reward,done,info=advance(branch,JointAction(None,action),'identity_check' if selfcheck else 'branch')
                assert done==(day==28);rr.append(float(reward))
                rt.append(dict(day=day,obs=obs,price_actions=price,actions=action,reward=float(reward),info=info))
                if hasattr(env,'sim'):
                    mask=(np.asarray(action)>0)&(np.asarray(trace[day-1]['actions'])>0)
                    assert np.array_equal(branch._last_lead_times[mask],lead_rows[day-1][mask]);common_pairs+=int(mask.sum())
                    if np.array_equal(info['price'],trace[day-1]['info']['price']):
                        assert np.array_equal(info['demand'],trace[day-1]['info']['demand']);same_price+=1
            assert stamp(snapshot)==before and stamp(env)==parent
            if selfcheck:
                assert rr==rewards[row['start']:] and serial(rt)==serial(trace[row['start']:])
                assert torch.equal(torch.get_rng_state(),terminal);selfchecks+=1
        assert rr==row['reference_rewards'] and trace_digest(rt)==saved['reference_trace_sha256']
        actual=np.asarray(rewards[row['start']:]);reference=np.asarray(rr)
        target=float(np.dot(model.gamma**np.arange(len(rr)),(actual-reference)/100))
        assert abs(target-row['target'])<1e-10
        if not identity:
            assert row['branch_days']==len(rr);branches+=len(rr)
            plan.append(dict(role=role,day=row['start']+1,days=len(rr)))
    torch.set_rng_state(terminal)
    assert item['reference_days']==branches and item['skipped_identical']==skips
    assert stamp(env)==parent
    targets={}
    whole=np.asarray(item['reference_rewards'] if arm=='standard' else item['whole_reference_rewards'])
    assert whole.shape==(28,)
    for role,rows in item['rows'].items():
        assert [r['start'] for r in rows]==([0,7,14,21] if role=='pricing' else list(range(28)))
        if arm=='coordinate':values=[r['target'] for r in rows]
        else:values=[sum(model.gamma**j*(a-b)/100 for j,(a,b) in enumerate(zip(rewards[r['start']:],whole[r['start']:]))) for r in rows]
        targets[role]=values
    return dict(targets=targets,branch_days=branches,skips=skips,selfchecks=selfchecks,common_pairs=common_pairs,
                same_price_pairs=same_price,branch_plan=plan)
