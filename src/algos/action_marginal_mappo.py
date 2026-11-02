"""Experimental action-marginal PPO, not used by prior frozen studies.

Same Gaussian cover sampling and physical execution, but PPO uses exact grid
action mass, integrating out latent values with identical executed orders.
Double-precision log-CDF differences protect narrow intervals and tail masses.
"""
import time
import numpy as np
import torch
from torch.distributions import Categorical
from src.algos.adaptive_structured_mappo import AdaptiveOrderingActor
from src.algos.value_normalized_mappo import ValueNormalizedMAPPO


class ActionMarginalOrderingActor(AdaptiveOrderingActor):
    def distribution(self,features):
        return super().distribution(features[...,:7])

    def grid_log_masses(self,features):
        if features.shape[-2:]!=(10,9) or not torch.isfinite(features).all():
            raise ValueError('finite10x9 local features required')
        if (features[...,7:]<0).any():raise ValueError('nonnegative position/forecast required')
        gaussian=self.distribution(features)
        mean=gaussian.mean.double()[...,None];sd=gaussian.scale.double()[...,None]
        position=features[...,7];forecast=features[...,8]
        # Match canonical float32 clipping/rounding when finding the final bin.
        maximum=torch.ceil(torch.clamp(20*forecast-position,min=0)/5).clamp(0,40).long()
        safe_forecast=torch.where(forecast>0,forecast,torch.ones_like(forecast)).double()[...,None]
        position=position.double()[...,None]
        index=torch.arange(41,dtype=torch.float64,device=features.device)
        low=((position+5*(index-1))/safe_forecast-mean)/sd
        high=((position+5*index)/safe_forecast-mean)/sd
        positive=low>0
        log_a=torch.where(positive,torch.special.log_ndtr(-low),torch.special.log_ndtr(high))
        log_b=torch.where(positive,torch.special.log_ndtr(-high),torch.special.log_ndtr(low))
        # Relative differences below double precision get a numerical-domain
        # guard. No probability floor is imposed on reachable action bins.
        difference=(log_b-log_a).clamp_max(-torch.finfo(torch.float64).eps)
        interior=log_a+torch.log(-torch.expm1(difference))
        masses=torch.where(index==0,torch.special.log_ndtr(high),interior)
        masses=torch.where(index==maximum[...,None],torch.special.log_ndtr(-low),masses)
        masses=torch.where(index<=maximum[...,None],masses,torch.full_like(masses,-torch.inf))
        deterministic=torch.where(index==0,torch.zeros_like(masses),torch.full_like(masses,-torch.inf))
        masses=torch.where((maximum==0)[...,None],deterministic,masses)
        return masses.to(features.dtype)

    def grid_distribution(self,features):
        return Categorical(logits=self.grid_log_masses(features))

    def stats(self,features,actions):
        if not torch.isfinite(actions).all() or not ((actions>=0)&(actions<=40)&(actions==actions.floor())).all():
            raise ValueError('integer grid actions0..40 required')
        distribution=self.grid_distribution(features)
        return distribution.log_prob(actions.long()).sum(-1),distribution.entropy().sum(-1)


class ActionMarginalMAPPO(ValueNormalizedMAPPO):
    def __init__(self,env,covers,seed=0,*,critic_mode='central',actor_lr=.003,critic_target='gae'):
        super().__init__(env,covers,seed,critic_mode=critic_mode,actor_lr=actor_lr,critic_target=critic_target)
        self.ordering=ActionMarginalOrderingActor(covers)
        self.actor_optimizers['ordering']=torch.optim.Adam(self.ordering.parameters(),lr=actor_lr)

    def collect(self,env,key,*,greedy=False,frozen=(),training=True,deadline=float('inf')):
        episode=super().collect(env,key,greedy=greedy,frozen=frozen,training=training,deadline=deadline)
        if training:
            with torch.no_grad():
                records=episode['rows']['ordering']
                for row,trace in zip(records,episode['trace']):
                    obs=np.asarray(trace['obs'],np.float32)
                    position=obs[:10]+obs[10:80].reshape(10,7).sum(1)
                    row['features']=np.concatenate([row['features'],position[:,None],obs[-10:,None]],axis=-1)
                    row['action']=np.asarray(trace['actions'],dtype=np.int64).copy()
                features=torch.as_tensor(np.asarray([r['features'] for r in records]),dtype=torch.float32)
                actions=torch.as_tensor(np.asarray([r['action'] for r in records]),dtype=torch.long)
                logp,_=self.ordering.stats(features,actions)
                for row,value in zip(records,logp):row['logp']=float(value)
        if time.perf_counter()>deadline:raise TimeoutError('Action-marginal collection deadline')
        return episode

    def checkpoint(self):
        data=super().checkpoint();data['variant']='action_marginal_structured_mappo_v1';return data

    def restore(self,data):
        if data['variant']!='action_marginal_structured_mappo_v1':raise ValueError('action-marginal checkpoint required')
        super().restore(dict(data,variant='value_normalized_structured_mappo_v1'))
