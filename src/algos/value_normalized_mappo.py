"""Structured adaptive MAPPO with normalized critic regression.

Actors, on-policy collection, cadence and GAE are inherited. Values passed to
GAE are always denormalized. Running moments are updated once per joint batch,
with equal pricing/ordering weight, and are part of the checkpoint. This uses
ValueNorm (not PopArt): the output layer is not rescaled when moments change.
"""
import numpy as np
import torch
from torch import nn
from src.algos.adaptive_structured_mappo import AdaptiveStructuredMAPPO
from src.algos.mappo import interval_gae


class RunningValueNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean_ema',torch.zeros((),dtype=torch.float64))
        self.register_buffer('square_ema',torch.zeros((),dtype=torch.float64))
        self.register_buffer('mass',torch.zeros((),dtype=torch.float64))
        self.register_buffer('updates',torch.zeros((),dtype=torch.int64))

    @torch.no_grad()
    def update(self,targets):
        if len(targets)!=2 or any(t.numel()==0 or not torch.isfinite(t).all() for t in targets):
            raise ValueError('two nonempty finite role targets required')
        mean=torch.stack([t.double().mean() for t in targets]).mean()
        square=torch.stack([t.double().square().mean() for t in targets]).mean()
        self.mean_ema.mul_(.99999).add_(mean*1e-5)
        self.square_ema.mul_(.99999).add_(square*1e-5)
        self.mass.mul_(.99999).add_(1e-5);self.updates.add_(1)

    def moments(self):
        if not int(self.updates):return self.mean_ema.float(),torch.ones((),dtype=torch.float32)
        mean=self.mean_ema/self.mass
        std=(self.square_ema/self.mass-mean.square()).clamp_min(.01).sqrt()
        return mean.float(),std.float()

    def normalize(self,value):
        mean,std=self.moments();return (value-mean)/std

    def denormalize(self,value):
        mean,std=self.moments();return value*std+mean


class ValueNormalizedMAPPO(AdaptiveStructuredMAPPO):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.critic_target!='gae':raise ValueError('normalized variant uses GAE targets')
        self.value_norm=RunningValueNorm()

    @torch.no_grad()
    def value(self,role,state):
        prediction=self.critic(role)(torch.as_tensor(state)).squeeze()
        return float(self.value_norm.denormalize(prediction))

    def update(self,episodes,*,epochs=4):
        if not episodes:raise ValueError('nonempty on-policy batch required')
        frozen=set(episodes[0]['frozen'])
        if any(set(ep['frozen'])!=frozen for ep in episodes):raise ValueError('mixed frozen roles')
        batches={};metrics={}
        for role in ('pricing','ordering'):
            records=[];advantages=[];returns=[]
            for ep in episodes:
                r=ep['rows'][role];starts=[v['start'] for v in r]
                if starts!=([0,7,14,21] if role=='pricing' else list(range(28))):
                    raise ValueError('invalid decision cadence')
                a,t=interval_gae(np.asarray(ep['rewards'])/100.,[v['value'] for v in r],starts,self.gamma,.95)
                records.extend(r);advantages.extend(a);returns.extend(t)
            tensor=lambda values:torch.as_tensor(np.asarray(values),dtype=torch.float32)
            features=tensor([r['features'] for r in records])
            actions=torch.as_tensor(np.asarray([r['action'] for r in records]),
                                    dtype=torch.long if role=='pricing' else torch.float32)
            old=tensor([r['logp'] for r in records]);adv=tensor(advantages)
            adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
            actor=getattr(self,role);steps=0
            if role not in frozen:
                for _ in range(epochs):
                    lp,_=actor.stats(features,actions);logratio=lp-old;ratio=logratio.exp()
                    if float(((ratio-1)-logratio).mean().detach())>.02:break
                    loss=-torch.minimum(ratio*adv,ratio.clamp(.9,1.1)*adv).mean()
                    optimizer=self.actor_optimizers[role]
                    optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),.5)
                    optimizer.step();steps+=1
            with torch.no_grad():
                lp,_=actor.stats(features,actions);logratio=lp-old
                metrics[role+'_kl']=float((logratio.exp()-1-logratio).mean())
            metrics[role+'_actor_steps']=steps
            batches[role]=(tensor([r['state'] for r in records]),tensor(returns))
            metrics[role+'_critic_target_mean']=float(np.mean(returns))
        self.value_norm.update([target for _,target in batches.values()])
        for _ in range(epochs):
            losses=[(self.critic(role)(s).squeeze(-1)-self.value_norm.normalize(t)).square().mean()
                    for role,(s,t) in batches.items()]
            loss=.25*sum(losses)
            self.critic_optimizer.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(self.critics.parameters(),.5);self.critic_optimizer.step()
        mean,std=self.value_norm.moments()
        metrics.update(critic_loss=float(loss.detach()),value_mean=float(mean),value_std=float(std),
                       value_norm_updates=int(self.value_norm.updates))
        if not all(np.isfinite(v) for v in metrics.values()):raise FloatingPointError('nonfinite update')
        return metrics

    def checkpoint(self):
        data=super().checkpoint();data['variant']='value_normalized_structured_mappo_v1';return data

    def restore(self,data):
        if data['variant']!='value_normalized_structured_mappo_v1':raise ValueError('normalized checkpoint required')
        super().restore(dict(data,variant='adaptive_structured_mappo_v1'))
