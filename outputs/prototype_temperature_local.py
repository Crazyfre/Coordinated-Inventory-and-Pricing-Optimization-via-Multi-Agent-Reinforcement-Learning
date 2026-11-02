"""UNINSTALLED local-critic control; no calibrated fitting on import."""
import torch
from outputs.temperature_integrated_mappo import TemperatureIntegratedMAPPO,TemperatureOrderingActor
from outputs.coordinate_reference_mappo import CoordinateReferenceMAPPO
from outputs.linear_coordinate_diagnostic import LinearPricingAdapter
from src.algos.categorical_residual_mappo import CategoricalResidualMAPPO


class TemperatureLocalPrototype(TemperatureIntegratedMAPPO):
    def __init__(self,env,covers,seed=0,*,arm):
        if arm not in ('fixed','annealed'):raise ValueError('declared arm required')
        # Initialize LOCAL critics from the start. Do not construct/discard a
        # central network or read global state even for its dimension.
        CoordinateReferenceMAPPO.__init__(self,env,covers,seed,critic_mode='local')
        original=CategoricalResidualMAPPO(env,covers,seed,critic_mode='local')
        self.pricing=LinearPricingAdapter(original.pricing,env.base_price)
        self.ordering=TemperatureOrderingActor(covers);self.temperature_arm=arm
        with torch.no_grad():
            self.pricing.actor.weight.zero_()
            self.pricing.actor.weight[:,0,:]=torch.log(torch.tensor([.005,.005,.005,.005,.98]))
            self.ordering.weight.zero_()
            self.ordering.weight[:,0,:]=torch.log(torch.tensor([.32,.36,.32]))
        self.actor_optimizers={r:torch.optim.Adam(getattr(self,r).parameters(),lr=.003) for r in ('pricing','ordering')}

    def checkpoint(self):
        data=super().checkpoint();data.update(variant='temperature_local_control_prototype_v1',
            critic_information='role_local_separate_v1');return data

    def restore(self,data):
        if (data.get('variant')!='temperature_local_control_prototype_v1' or data.get('mode')!='local'
                or data.get('critic_information')!='role_local_separate_v1'):
            raise ValueError('matching local-temperature control schema required')
        super().restore(dict(data,variant='temperature_integrated_mappo_v1'))
