"""Experimental joint neural residual PPO; original model/rewards unchanged.

Both actors retain their existing LOCAL observations and public day, encoded by
normalized64x64 MLPs. Zero output heads preserve the exact categorical warmstart.
Shared central critic, full-return/reference update and physical grid mass reuse
the already tested residual algorithm. No calibrated job imports this prototype.
"""
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from src.algos.categorical_residual_mappo import CategoricalResidualMAPPO,ResidualOrderingActor
from src.algos.structured_pricing_ppo import critic_features
from src.algos.parameter_policy_gradient import order_action
from src.env.interface import JointAction


class NormalizedLocalMLP(nn.Module):
    def __init__(self,width,prior):
        super().__init__();self.width=int(width)
        self.input_norm=nn.LayerNorm(width)
        self.fc1=nn.Linear(width,64);self.norm1=nn.LayerNorm(64)
        self.fc2=nn.Linear(64,64);self.norm2=nn.LayerNorm(64)
        self.output=nn.Linear(64,10*len(prior))
        nn.init.zeros_(self.output.weight)
        with torch.no_grad():self.output.bias.copy_(torch.log(torch.tensor(prior)).repeat(10))

    def forward(self,x):
        x=self.input_norm(x);x=self.norm1(torch.tanh(self.fc1(x)))
        return self.output(self.norm2(torch.tanh(self.fc2(x))))


class NeuralPricingActor(nn.Module):
    def __init__(self,width):
        super().__init__();self.body=NormalizedLocalMLP(width,[.005,.005,.005,.005,.98])

    def features(self,obs,day):return critic_features(obs,day)

    def distribution(self,features):
        logits=self.body(features)
        return Categorical(logits=logits.reshape(logits.shape[:-1]+(10,5)))

    def stats(self,features,actions):
        d=self.distribution(features);return d.log_prob(actions.long()).sum(-1),d.entropy().sum(-1)

    @torch.no_grad()
    def act(self,features,greedy=False):
        d=self.distribution(torch.as_tensor(features));q=d.probs.argmax(-1) if greedy else d.sample()
        return q.numpy(),float(d.log_prob(q).sum())


class NeuralOrderingActor(ResidualOrderingActor):
    def __init__(self,covers):
        super().__init__(covers);del self.weight
        self.body=NormalizedLocalMLP(271,[.05,.9,.05])
        selector=torch.zeros(10,21,3)
        for k in range(3):selector[:,k,k]=1
        self.register_buffer('_mass_selector',selector)

    def features(self,obs,day):
        anchor=order_action(obs,day,self.covers.detach().numpy())
        return np.r_[critic_features(obs,day),anchor].astype(np.float32)

    def grid_log_masses(self,features):
        if features.shape[-1]!=271 or not torch.isfinite(features).all():
            raise ValueError('finite271local features required')
        anchor=features[...,-10:]
        inputs=torch.cat((features[...,:261],anchor/40),dim=-1)
        logits=self.body(inputs);logits=logits.reshape(logits.shape[:-1]+(10,3))
        zeros=torch.zeros(logits.shape[:-1]+(18,),dtype=logits.dtype,device=logits.device)
        canonical=torch.cat((logits,zeros,anchor[...,None]),dim=-1)
        class Integral:
            pass
        integral=Integral();integral.weight=self._mass_selector
        # Fixed selector makes the tested affine-logit mass integral consume
        # these neural logits directly, including clipping/duplicate-bin sums.
        return ResidualOrderingActor.grid_log_masses(integral,canonical)


class NeuralResidualMAPPO(CategoricalResidualMAPPO):
    def __init__(self,env,covers,seed=0,*,critic_mode='central'):
        super().__init__(env,covers,seed,critic_mode=critic_mode)
        self.pricing=NeuralPricingActor(len(env.pricing_observation())+1)
        self.ordering=NeuralOrderingActor(covers)
        self.actor_optimizers={r:torch.optim.Adam(getattr(self,r).parameters(),lr=.003) for r in ('pricing','ordering')}

    def collect(self,env,key,*,greedy=False,frozen=(),training=True,deadline=float('inf')):
        if not set(frozen)<= {'pricing','ordering'}:raise ValueError('unknown frozen actor')
        env.reset(seed=key);rows={'pricing':[],'ordering':[]};rewards=[];trace=[]
        for day in range(1,29):
            if time.perf_counter()>deadline:raise TimeoutError('neural residual MAPPO deadline')
            price=None
            if env.pricing_due:
                obs=env.pricing_observation().copy();f=self.pricing.features(obs,day)
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

    def checkpoint(self):
        data=super().checkpoint();data.update(variant='neural_residual_mc_mappo_v1',encoder='local_layernorm_mlp64x2');return data

    def restore(self,data):
        if data['variant']!='neural_residual_mc_mappo_v1' or data['encoder']!='local_layernorm_mlp64x2':
            raise ValueError('neural residual checkpoint required')
        super().restore(dict(data,variant='categorical_residual_mc_mappo_v1'))
