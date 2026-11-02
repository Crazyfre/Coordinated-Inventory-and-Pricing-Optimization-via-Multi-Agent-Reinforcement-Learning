"""New frozen-prebatch Q collector, no creation/fit on import."""
import copy
import numpy as np
import torch
from outputs.coordinate_reference_mappo import clone_local_env
from outputs.enumerated_credit_diagnostic import trace_hash
from src.algos.parameter_policy_gradient import order_action
from src.env.interface import JointAction

def active_collect(self,env,key,*,greedy=False,frozen=(),training=True,deadline=float('inf'),
            budget=None,keep_proof_snapshots=False):
    if not training:raise ValueError('training collector only')
    if greedy or frozen:raise ValueError('prototype requires both stochastic learning actors')
    if budget is None:raise ValueError('explicit synthetic proof step budget required')
    env.reset(seed=key);coordinate_rng=np.random.default_rng(np.random.SeedSequence([key,6179]))
    rows={'pricing':[],'ordering':[]};snapshots=[];rewards=[];trace=[]
    def record(role,obs,features,actions,coordinate,reference_action,multiplier):
        state=self.indexed_state(env,obs,env.day,role,coordinate)
        lp=self._scores(role,features,actions)
        row=dict(features=features,action=actions.copy(),coordinate=coordinate,coordinate_multiplier=multiplier,
            component_logps=lp.copy(),logp=float(lp[coordinate]),joint_logp=float(lp.sum()),
            state=state,value=self.value(role,state),q_pre=self.q_table(state,role,features),start=len(rewards),
            reference_action=reference_action.copy())
        rows[role].append(row)
        # After root action sampling, before applying that action. This RNG
        # state belongs to future choices and is restored independently.
        snapshots.append((role,row,clone_local_env(env),torch.get_rng_state().clone()))
    for day in range(1,29):
        price=None
        if env.pricing_due:
            obs=env.pricing_observation().copy();f=self.pricing.features(obs,day)
            eligible=np.arange(10)
            multiplier=len(eligible);coordinate=int(eligible[int(coordinate_rng.integers(multiplier))]) if multiplier else 0
            price,_=self.pricing.act(f);reference=price.copy();reference[coordinate]=4
            record('pricing',obs,f,price,coordinate,reference,multiplier);env.select_price(price)
        coordinate=int(coordinate_rng.integers(10))
        obs=env.replenishment_observation().copy();f=self.ordering.features(obs,day)
        action,_=self.ordering.act(f)
        anchor=order_action(obs,day,self.ordering.covers.detach().numpy())
        reference=action.copy();reference[coordinate]=anchor[coordinate]
        record('ordering',obs,f,action,coordinate,reference,10)
        budget.claim('actual',deadline)
        _,reward,done,info=env.step(JointAction(None,action));assert done==(day==28)
        rewards.append(float(reward));trace.append(dict(day=day,obs=obs,price_actions=price,
            actions=action,reward=float(reward),info=info))
    actual_rng=torch.get_rng_state().clone();reference_days=0;skipped=0
    try:
        for role,row,snapshot,rng in snapshots:
            if np.array_equal(row['action'],row['reference_action']):
                row['reference_rewards']=rewards[row['start']:].copy();row['branch_days']=0;skipped+=1
            else:
                torch.set_rng_state(rng)
                rr,rt=self.continuation(clone_local_env(snapshot),role,row['reference_action'],budget=budget,deadline=deadline)
                row['reference_rewards']=rr;row['branch_days']=len(rr);reference_days+=len(rr)
            actual=sum(self.gamma**j*r for j,r in enumerate(rewards[row['start']:]))/100
            baseline=sum(self.gamma**j*r for j,r in enumerate(row['reference_rewards']))/100
            row['target']=float(actual-baseline)
    finally:torch.set_rng_state(actual_rng)
    result=dict(rows=rows,rewards=rewards,trace=trace,frozen=[],reference_days=reference_days,
                skipped_identical=skipped,actual_policy_rng=actual_rng)
    if keep_proof_snapshots:result['proof_snapshots']=snapshots
    return result

def integrated_collect(self,env,key,*,budget,deadline):
    hashes={};original=self.continuation;had_override='continuation' in self.__dict__
    def observed(snapshot,role,action,**kwargs):
        first=snapshot.day;rr,trace=original(snapshot,role,action,**kwargs)
        hashes[role,first,tuple(int(x) for x in action)]=trace_hash(trace);return rr,trace
    self.continuation=observed
    try:
        ep=active_collect(self,env,key,budget=budget,deadline=deadline,keep_proof_snapshots=True)
        terminal=ep['actual_policy_rng'].clone();roots=[]
        try:
            for role,row,snapshot,rng in ep.pop('proof_snapshots'):
                i=row['coordinate'];actor=getattr(self,role)
                with torch.no_grad():p=actor.distribution(torch.tensor(row['features'])).probs[i].double().numpy()
                support=np.flatnonzero(p>0);entries=[];start=row['start'];first=start+1
                for value in support:
                    action=row['action'].copy();action[i]=value
                    if np.array_equal(action,row['action']):
                        rr=ep['rewards'][start:].copy();digest=trace_hash(ep['trace'][start:]);source='actual'
                    elif np.array_equal(action,row['reference_action']):
                        rr=row['reference_rewards'].copy();digest=hashes[role,first,tuple(int(x) for x in action)];source='strong'
                    else:
                        torch.set_rng_state(rng)
                        rr,trace=self.continuation(copy.deepcopy(snapshot),role,action,budget=budget,deadline=deadline)
                        digest=trace_hash(trace);source='branch'
                    entries.append(dict(action=int(value),probability=float(p[value]),rewards=rr,trace_sha256=digest,source=source,
                        discounted_return=float(np.dot(self.gamma**np.arange(len(rr)),rr)/100)))
                anchor=int(row['reference_action'][i]);base=next(r['discounted_return'] for r in entries if r['action']==anchor)
                for entry in entries:entry['residual_return']=entry['discounted_return']-base
                roots.append(dict(role=role,root_rng=rng.clone(),entries=entries))
        finally:torch.set_rng_state(terminal)
        ep['roots']=roots;return ep
    finally:
        if had_override:self.continuation=original
        else:del self.continuation
