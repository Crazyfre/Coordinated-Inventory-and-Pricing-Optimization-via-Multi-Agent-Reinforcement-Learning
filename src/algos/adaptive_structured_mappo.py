"""MAPPO with trainable, state-independent ordering dispersion.

Only10 log-standard-deviation parameters differ from the fixed-noise variant.
Pricing, mean cover structure, local observations, critic and GAE remain as in
StructuredMAPPO. Original checkpoint schemas are deliberately kept separate.
"""
import math
import torch
from torch import nn
from torch.distributions import Normal
from src.algos.structured_ordering_ppo import StructuredOrderingActor
from src.algos.structured_mappo import StructuredMAPPO


class AdaptiveOrderingActor(StructuredOrderingActor):
    def __init__(self,covers):
        super().__init__(covers)
        self.log_std=nn.Parameter(torch.full((10,),math.log(.75)))

    def distribution(self,features):
        mean=self.covers+(features*self.weight).sum(-1)
        return Normal(mean,self.log_std.clamp(-3.,.5).exp())


class AdaptiveStructuredMAPPO(StructuredMAPPO):
    def __init__(self,env,covers,seed=0,*,critic_mode='central',actor_lr=.003,critic_target='gae'):
        super().__init__(env,covers,seed,critic_mode=critic_mode,actor_lr=actor_lr,critic_target=critic_target)
        self.ordering=AdaptiveOrderingActor(covers)
        self.actor_optimizers['ordering']=torch.optim.Adam(self.ordering.parameters(),lr=actor_lr)

    def checkpoint(self):
        data=super().checkpoint();data['variant']='adaptive_structured_mappo_v1'
        return data

    def restore(self,data):
        if data['variant']!='adaptive_structured_mappo_v1':raise ValueError('adaptive checkpoint required')
        compatible=dict(data,variant='structured_mappo_v1')
        super().restore(compatible)
