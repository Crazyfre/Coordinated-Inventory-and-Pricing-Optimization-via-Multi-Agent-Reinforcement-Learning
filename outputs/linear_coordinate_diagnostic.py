"""Diagnostic-only adapter: exact original linear actors, no optimizer updates.

Preserves the old proof-bound coordinate collector and production algorithm.
The initializer may only create a fresh untrained policy; restore/update reject.
"""
import copy
import numpy as np
import torch
from torch import nn
from outputs.coordinate_reference_mappo import CoordinateReferenceMAPPO
from src.algos.categorical_residual_mappo import CategoricalResidualMAPPO
from src.algos.structured_pricing_ppo import pricing_features


class LinearPricingAdapter(nn.Module):
    def __init__(self,actor,base_prices):
        super().__init__();self.actor=copy.deepcopy(actor)
        self.register_buffer('base_prices',torch.as_tensor(np.asarray(base_prices).copy(),dtype=torch.float32))
    def features(self,obs,day):return pricing_features(obs,day,self.base_prices.detach().numpy())
    def distribution(self,features):return self.actor.distribution(features)
    def stats(self,features,actions):return self.actor.stats(features,actions)
    def act(self,features,greedy=False):return self.actor.act(features,greedy)


class LinearCoordinateDiagnostic(CoordinateReferenceMAPPO):
    def __init__(self,env,covers,seed=0):
        super().__init__(env,covers,seed,critic_mode='central')
        original=CategoricalResidualMAPPO(env,covers,seed)
        self.pricing=LinearPricingAdapter(original.pricing,env.base_price)
        self.ordering=copy.deepcopy(original.ordering)
        self.actor_optimizers={}
        assert sum(p.numel() for p in self.pricing.parameters())==350
        assert sum(p.numel() for p in self.ordering.parameters())==630
    def update(self,*args,**kwargs):raise RuntimeError('diagnostic only: zero updates permitted')
    def restore(self,*args,**kwargs):raise RuntimeError('diagnostic only: no checkpoint adoption or resume')
    def checkpoint(self):
        return dict(variant='linear_coordinate_signal_diagnostic_v1',gamma=self.gamma,
            trainable=False,updates=0,model=self.state_dict())


def episode_gradients(model,episode,whole_reference):
    """Raw old-policy score estimates; no clipping/empirical normalization.

Includes gamma**start and role-mean weighting for BOTH estimators. The latter
scales each role's gradient equally within that role; it does not claim a single
unweighted joint-role learning objective or actual clipped-PPO equivalence.
"""
    result={}
    for role in ('pricing','ordering'):
        rows=episode['rows'][role];starts=[row['start'] for row in rows]
        if starts!=([0,7,14,21] if role=='pricing' else list(range(28))):raise ValueError('wrong role cadence')
        if any(row['value']!=0. for row in rows):raise ValueError('zero untrained critic required')
        actual=np.asarray(episode['rewards'],dtype=float);baseline=np.asarray(whole_reference,dtype=float)
        if actual.shape!=(28,) or baseline.shape!=(28,):raise ValueError('28day actual/reference required')
        target=[]
        for row in rows:
            t=row['start'];n=28-t
            target.append(np.dot(model.gamma**np.arange(n),(actual[t:]-baseline[t:])/100))
        features=torch.as_tensor(np.asarray([row['features'] for row in rows]),dtype=torch.float32)
        actions=torch.as_tensor(np.asarray([row['action'] for row in rows]),dtype=torch.long)
        chosen=torch.tensor([row['coordinate'] for row in rows],dtype=torch.long)
        time_weight=torch.tensor([model.gamma**t for t in starts],dtype=torch.float32)
        ordinary_target=torch.tensor(target,dtype=torch.float32)
        coordinate_target=torch.tensor([row['target'] for row in rows],dtype=torch.float32)
        actor=getattr(model,role);parameters=list(actor.parameters());assert len(parameters)==1
        scores=actor.distribution(features).log_prob(actions)
        ordinary=(scores.sum(-1)*ordinary_target*time_weight).mean()
        coordinate=(10*scores.gather(1,chosen[:,None]).squeeze(1)*coordinate_target*time_weight).mean()
        g=torch.autograd.grad(ordinary,parameters,retain_graph=True)[0]
        h=torch.autograd.grad(coordinate,parameters)[0]
        assert torch.isfinite(g).all() and torch.isfinite(h).all()
        result[role]=dict(ordinary=g.detach().double().flatten().numpy(),coordinate=h.detach().double().flatten().numpy())
    return result


def summarize(gradients):
    result={}
    for role in ('pricing','ordering'):
        arrays={kind:np.stack([row[role][kind] for row in gradients]) for kind in ('ordinary','coordinate')}
        stats={}
        for kind,values in arrays.items():
            mean=values.mean(0);centered=values-mean
            variance=float(np.square(centered).sum()/(len(values)-1))
            norm=float(np.dot(mean,mean))
            stats[kind]=dict(mean=mean.tolist(),mean_squared_norm=norm,centered_trace_variance=variance,
                mean_episode_squared_norm=float(np.square(values).sum(1).mean()),
                estimated_mean_snr=norm/(variance/len(values)) if variance>0 else None)
        x=arrays['ordinary'].mean(0);y=arrays['coordinate'].mean(0);den=np.linalg.norm(x)*np.linalg.norm(y)
        stats['coordinate_over_ordinary_variance']=(stats['coordinate']['centered_trace_variance']/stats['ordinary']['centered_trace_variance']
            if stats['ordinary']['centered_trace_variance']>0 else None)
        stats['estimated_mean_direction_cosine']=float(np.dot(x,y)/den) if den>0 else None
        result[role]=stats
    return result
