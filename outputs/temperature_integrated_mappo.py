"""New controlled ordering-temperature variant; original frozen classes untouched."""
import math
import torch
from torch import nn
from outputs.anchored_pricing_mappo import AnchoredPricingIntegratedMAPPO
from src.algos.categorical_residual_mappo import ResidualOrderingActor


class TemperatureOrderingActor(ResidualOrderingActor):
    def __init__(self,covers):
        super().__init__(covers);self.register_buffer('temperature',torch.tensor(1.,dtype=torch.float32))

    def set_temperature(self,value):
        value=float(value)
        if not math.isfinite(value) or not .25<=value<=1:raise ValueError('temperature must be in[.25,1]')
        self.temperature.fill_(value)

    def grid_log_masses(self,features):
        if features.shape[-2:]!=(10,22) or not torch.isfinite(features).all():raise ValueError('finite10x22features required')
        if not .25<=float(self.temperature)<=1:raise ValueError('invalid temperature buffer')
        anchor=features[...,-1]
        if not ((anchor==anchor.floor())&(anchor>=0)&(anchor<=40)).all():raise ValueError('integer anchor required')
        logits=(torch.einsum('...if,ifk->...ik',features[...,:21],self.weight)/self.temperature).log_softmax(-1)
        physical=(anchor[...,None]+torch.tensor([-1,0,1],device=features.device)).clamp(0,40)
        mask=physical[...,None,:]==torch.arange(41,device=features.device)[...,None];reachable=mask.any(-1)
        values=torch.where(mask,logits[...,None,:],torch.full_like(logits[...,None,:],-torch.inf))
        values=torch.where(reachable[...,None],values,torch.zeros_like(values))
        return torch.where(reachable,values.logsumexp(-1),torch.full_like(anchor[...,None].expand_as(reachable),-torch.inf))


class TemperatureIntegratedMAPPO(AnchoredPricingIntegratedMAPPO):
    def __init__(self,env,covers,seed=0,*,arm):
        if arm not in ('fixed','annealed'):raise ValueError('declared temperature arm required')
        super().__init__(env,covers,seed);self.temperature_arm=arm
        previous=self.ordering;self.ordering=TemperatureOrderingActor(covers)
        with torch.no_grad():self.ordering.weight.copy_(previous.weight)
        self.actor_optimizers['ordering']=torch.optim.Adam(self.ordering.parameters(),lr=.003)

    def checkpoint(self):
        data=super().checkpoint();data.update(variant='temperature_integrated_mappo_v1',
            temperature_arm=self.temperature_arm,ordering_temperature=float(self.ordering.temperature),
            temperature_design='latent_logits_before_physical_merge_v1')
        return data

    def restore(self,data):
        tau=data.get('ordering_temperature');saved=data.get('model',{}).get('ordering.temperature')
        if (data.get('variant')!='temperature_integrated_mappo_v1' or data.get('temperature_arm')!=self.temperature_arm
                or data.get('temperature_design')!='latent_logits_before_physical_merge_v1'
                or not isinstance(tau,(float,int)) or not math.isfinite(tau) or not .25<=tau<=1
                or not isinstance(saved,torch.Tensor) or saved.numel()!=1 or float(saved)!=tau):
            raise ValueError('matching temperature schema/arm/value required')
        super().restore(dict(data,variant='anchored_pricing_integrated_mappo_v1'))
