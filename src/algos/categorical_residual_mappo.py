"""Untrained prototype: physical discrete order corrections + full-return PPO.

Original economics, local actors, weekly/daily cadence and central training-only
critic. Combined candidate, not an isolated attribution for any one component.
This module is imported by no frozen calibrated study.
"""
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from src.algos.structured_mappo import StructuredMAPPO
from src.algos.value_normalized_mappo import RunningValueNorm
from src.algos.pipeline_structured_mappo import pipeline_features
from src.algos.structured_pricing_ppo import pricing_features
from src.algos.parameter_policy_gradient import order_action
from src.env.interface import JointAction


class ResidualOrderingActor(nn.Module):
    def __init__(self,covers):
        super().__init__();c=torch.as_tensor(covers,dtype=torch.float32)
        if c.shape!=(10,) or not torch.isfinite(c).all():raise ValueError('ten finite anchor covers required')
        self.register_buffer('covers',c.clone());self.weight=nn.Parameter(torch.zeros(10,21,3))
        with torch.no_grad():self.weight[:,0,:]=torch.log(torch.tensor([.05,.9,.05]))

    def features(self,obs,day):
        anchor=order_action(obs,day,self.covers.detach().numpy())
        return np.c_[pipeline_features(obs,day,'timing'),anchor].astype(np.float32)

    def grid_log_masses(self,features):
        if features.shape[-2:]!=(10,22) or not torch.isfinite(features).all():
            raise ValueError('finite10x22 local features required')
        anchor=features[...,-1]
        if not ((anchor==anchor.floor())&(anchor>=0)&(anchor<=40)).all():raise ValueError('integer anchor0..40 required')
        logits=torch.einsum('...if,ifk->...ik',features[...,:21],self.weight).log_softmax(-1)
        physical=(anchor[...,None]+torch.tensor([-1,0,1],device=features.device)).clamp(0,40)
        mask=physical[...,None,:]==torch.arange(41,device=features.device)[...,None]
        reachable=mask.any(-1)
        values=torch.where(mask,logits[...,None,:],torch.full_like(logits[...,None,:],-torch.inf))
        # Give invalid bins a finite internal logsumexp, then restore -inf.
        # This prevents an all--inf backward path from producing0*NaN gradients.
        values=torch.where(reachable[...,None],values,torch.zeros_like(values))
        return torch.where(reachable,values.logsumexp(-1),torch.full_like(anchor[...,None].expand_as(reachable),-torch.inf))

    def distribution(self,features):return Categorical(logits=self.grid_log_masses(features))

    def stats(self,features,actions):
        if not torch.isfinite(actions).all() or not ((actions==actions.floor())&(actions>=0)&(actions<=40)).all():
            raise ValueError('integer physical action0..40 required')
        distribution=self.distribution(features)
        return distribution.log_prob(actions.long()).sum(-1),distribution.entropy().sum(-1)

    @torch.no_grad()
    def act(self,features,greedy=False):
        d=self.distribution(torch.as_tensor(features));q=d.probs.argmax(-1) if greedy else d.sample()
        return q.numpy(),float(d.log_prob(q).sum())


def full_returns(rewards,gamma):
    values=np.zeros(len(rewards));tail=0.
    for t in range(len(rewards)-1,-1,-1):tail=float(rewards[t])+gamma*tail;values[t]=tail
    return values


class CategoricalResidualMAPPO(StructuredMAPPO):
    def __init__(self,env,covers,seed=0,*,critic_mode='central'):
        super().__init__(env,covers,seed,critic_mode=critic_mode,critic_target='mc')
        self.ordering=ResidualOrderingActor(covers)
        self.actor_optimizers['ordering']=torch.optim.Adam(self.ordering.parameters(),lr=.003)
        self.value_norm=RunningValueNorm()

    @torch.no_grad()
    def value(self,role,state):
        return float(self.value_norm.denormalize(self.critic(role)(torch.as_tensor(state)).squeeze()))

    def collect(self,env,key,*,greedy=False,frozen=(),training=True,deadline=float('inf')):
        if not set(frozen)<= {'pricing','ordering'}:raise ValueError('unknown frozen actor')
        env.reset(seed=key);rows={'pricing':[],'ordering':[]};rewards=[];trace=[]
        for day in range(1,29):
            if time.perf_counter()>deadline:raise TimeoutError('residual MAPPO deadline')
            price=None
            if env.pricing_due:
                obs=env.pricing_observation().copy();f=pricing_features(obs,day,env.base_price)
                price,lp=self.pricing.act(f,greedy or 'pricing' in frozen)
                if training:
                    state=self.critic_input(env,obs,day,'pricing')
                    rows['pricing'].append(dict(features=f,action=price.copy(),logp=lp,state=state,
                        value=self.value('pricing',state),start=len(rewards)))
                env.select_price(price)
            obs=env.replenishment_observation().copy();f=self.ordering.features(obs,day)
            action,lp=self.ordering.act(f,greedy or 'ordering' in frozen)
            if training:
                state=self.critic_input(env,obs,day,'ordering')
                rows['ordering'].append(dict(features=f,action=action.copy(),logp=lp,state=state,
                    value=self.value('ordering',state),start=len(rewards)))
            _,reward,done,info=env.step(JointAction(None,action));assert done==(day==28)
            rewards.append(float(reward));trace.append(dict(day=day,obs=obs,price_actions=price,
                actions=action,reward=float(reward),info=info))
        return dict(rows=rows,rewards=rewards,trace=trace,frozen=list(frozen))

    def update(self,episodes,*,epochs=4):
        if not episodes:raise ValueError('nonempty on-policy batch required')
        frozen=set(episodes[0]['frozen']);batches={};metrics={}
        if any(set(ep['frozen'])!=frozen for ep in episodes):raise ValueError('mixed frozen roles')
        targets=[]
        for ep in episodes:
            r=np.asarray(ep['rewards'],dtype=float);b=np.asarray(ep['reference_rewards'],dtype=float)
            if r.shape!=(28,) or b.shape!=(28,) or not np.isfinite(r).all() or not np.isfinite(b).all():
                raise ValueError('28finite actual/reference rewards required')
            targets.append(full_returns((r-b)/100,self.gamma))
        for role in ('pricing','ordering'):
            records=[];returns=[]
            for ep,t in zip(episodes,targets):
                rows=ep['rows'][role];starts=[r['start'] for r in rows]
                if starts!=([0,7,14,21] if role=='pricing' else list(range(28))):raise ValueError('invalid decision cadence')
                records.extend(rows);returns.extend(t[starts])
            tensor=lambda a:torch.as_tensor(np.asarray(a),dtype=torch.float32)
            f=tensor([r['features'] for r in records]);actions=torch.as_tensor(np.asarray([r['action'] for r in records]),dtype=torch.long)
            old=tensor([r['logp'] for r in records]);target=tensor(returns)
            adv=target-tensor([r['value'] for r in records]);adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
            actor=getattr(self,role);steps=0
            if role not in frozen:
                for _ in range(epochs):
                    lp,_=actor.stats(f,actions);lr=lp-old;ratio=lr.exp()
                    if float((ratio-1-lr).mean().detach())>.02:break
                    loss=-torch.minimum(ratio*adv,ratio.clamp(.9,1.1)*adv).mean()
                    optimizer=self.actor_optimizers[role];optimizer.zero_grad();loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(),.5);optimizer.step();steps+=1
            with torch.no_grad():
                lp,_=actor.stats(f,actions);lr=lp-old;metrics[role+'_kl']=float((lr.exp()-1-lr).mean())
            metrics[role+'_actor_steps']=steps;metrics[role+'_critic_target_mean']=float(target.mean())
            batches[role]=(tensor([r['state'] for r in records]),target)
        self.value_norm.update([t for _,t in batches.values()])
        for _ in range(epochs):
            loss=.25*sum((self.critic(role)(s).squeeze(-1)-self.value_norm.normalize(t)).square().mean()
                for role,(s,t) in batches.items())
            self.critic_optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.critics.parameters(),.5)
            self.critic_optimizer.step()
        mean,std=self.value_norm.moments()
        metrics.update(critic_loss=float(loss.detach()),value_mean=float(mean),value_std=float(std),value_norm_updates=int(self.value_norm.updates))
        if not all(np.isfinite(v) for v in metrics.values()):raise FloatingPointError('nonfinite update')
        return metrics

    def checkpoint(self):
        data=super().checkpoint();data['variant']='categorical_residual_mc_mappo_v1';return data

    def restore(self,data):
        if data['variant']!='categorical_residual_mc_mappo_v1':raise ValueError('categorical residual MC checkpoint required')
        super().restore(dict(data,variant='structured_mappo_v1'))
