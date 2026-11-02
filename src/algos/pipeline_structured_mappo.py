"""Isolated timing-representation experiment; original frozen algorithms intact.

Only local replenishment observations and the public day enter actor features.
The collapsed arm retains early/late totals but masks within-group timing.
Execution, pushforward likelihood, GAE, and normalized central critic are shared.
"""
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from src.algos.action_marginal_mappo import ActionMarginalOrderingActor
from src.algos.value_normalized_mappo import ValueNormalizedMAPPO
from src.algos.structured_ordering_ppo import ordering_features,execute_cover
from src.algos.structured_pricing_ppo import pricing_features
from src.env.interface import JointAction

MODES=('current','collapsed','timing')


def pipeline_features(obs,day,mode):
    if mode not in MODES:raise ValueError('unknown pipeline representation')
    base=ordering_features(obs,day)
    if mode=='current':return base
    x=np.asarray(obs,np.float32);pipe=x[10:80].reshape(10,7).copy()
    aggregate=pipe.sum(0)
    if mode=='collapsed':
        pipe[:,:3]=pipe[:,:3].mean(1,keepdims=True)
        pipe[:,3:]=pipe[:,3:].mean(1,keepdims=True)
        aggregate[:3]=aggregate[:3].mean();aggregate[3:]=aggregate[3:].mean()
    own=np.clip(pipe/(1+x[-10:,None])/20,0,2)
    congestion=np.broadcast_to(np.clip(aggregate/500,0,2),(10,7))
    return np.concatenate((base,own,congestion),axis=-1).astype(np.float32)


class PipelineOrderingActor(ActionMarginalOrderingActor):
    def __init__(self,covers,mode):
        super().__init__(covers)
        if mode not in MODES:raise ValueError('unknown pipeline representation')
        self.mode=mode;self.width=7 if mode=='current' else 21
        self.weight=nn.Parameter(torch.zeros(10,self.width))

    def distribution(self,features):
        return Normal(self.covers+(features[...,:self.width]*self.weight).sum(-1),
                      self.log_std.clamp(-3.,.5).exp())

    def grid_log_masses(self,features):
        # Reuse the tested exact pushforward integral. The adapter passes its
        # Gaussian unchanged and the existing integral consumes only the final
        # position/forecast columns of its canonical nine-column input.
        if features.shape[-2:]!=(10,self.width+2) or not torch.isfinite(features).all():
            raise ValueError('finite local mean features plus position/forecast required')
        gaussian=self.distribution(features)
        canonical=torch.cat((features[...,:7],features[...,-2:]),dim=-1)
        class Integral:
            def distribution(self,unused):return gaussian
        return ActionMarginalOrderingActor.grid_log_masses(Integral(),canonical)


class PipelineMAPPO(ValueNormalizedMAPPO):
    def __init__(self,env,covers,seed=0,*,representation='timing',critic_mode='central'):
        super().__init__(env,covers,seed,critic_mode=critic_mode)
        self.representation=representation
        self.ordering=PipelineOrderingActor(covers,representation)
        self.actor_optimizers['ordering']=torch.optim.Adam(self.ordering.parameters(),lr=.003)

    def collect(self,env,key,*,greedy=False,frozen=(),training=True,deadline=float('inf')):
        if not set(frozen)<= {'pricing','ordering'}:raise ValueError('unknown frozen actor')
        env.reset(seed=key);rows={'pricing':[],'ordering':[]};rewards=[];trace=[]
        for day in range(1,29):
            if time.perf_counter()>deadline:raise TimeoutError('pipeline MAPPO deadline')
            price=None
            if env.pricing_due:
                obs=env.pricing_observation().copy();f=pricing_features(obs,day,env.base_price)
                price,lp=self.pricing.act(f,greedy or 'pricing' in frozen)
                if training:
                    state=self.critic_input(env,obs,day,'pricing')
                    rows['pricing'].append(dict(features=f,action=price.copy(),logp=lp,
                        state=state,value=self.value('pricing',state),start=len(rewards)))
                env.select_price(price)
            obs=env.replenishment_observation().copy()
            f=pipeline_features(obs,day,self.representation)
            latent,_=self.ordering.act(f,greedy or 'ordering' in frozen)
            action=execute_cover(obs,latent)
            if training:
                position=obs[:10]+obs[10:80].reshape(10,7).sum(1)
                density=np.concatenate((f,position[:,None],obs[-10:,None]),axis=-1)
                state=self.critic_input(env,obs,day,'ordering')
                rows['ordering'].append(dict(features=density,action=action.copy(),
                    state=state,value=self.value('ordering',state),start=len(rewards)))
            _,reward,done,info=env.step(JointAction(None,action))
            if done!=(day==28):raise ValueError('complete28day episode required')
            rewards.append(float(reward))
            trace.append(dict(day=day,obs=obs,price_actions=price,actions=action,reward=float(reward),info=info))
        if training:
            with torch.no_grad():
                records=rows['ordering']
                f=torch.as_tensor(np.asarray([r['features'] for r in records]),dtype=torch.float32)
                a=torch.as_tensor(np.asarray([r['action'] for r in records]),dtype=torch.long)
                lp,_=self.ordering.stats(f,a)
                for row,v in zip(records,lp):row['logp']=float(v)
        if time.perf_counter()>deadline:raise TimeoutError('pipeline MAPPO deadline')
        return dict(rows=rows,rewards=rewards,trace=trace,frozen=list(frozen))

    def checkpoint(self):
        data=super().checkpoint()
        data.update(variant='pipeline_structured_mappo_v1',representation=self.representation)
        return data

    def restore(self,data):
        if data['variant']!='pipeline_structured_mappo_v1' or data['representation']!=self.representation:
            raise ValueError('matching pipeline representation checkpoint required')
        super().restore(dict(data,variant='value_normalized_structured_mappo_v1'))
