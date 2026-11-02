"""Uninstalled prospective prior-package controls, no calibrated fit.

Softer STOCHASTIC priors preserve the exact strong GREEDY initial pair. This is
an explicitly different initial distribution, not the frozen default prior.
"""
import torch
from outputs.linear_conditional_credit_mappo import LinearConditionalCreditMAPPO
from outputs.support_integrated_mappo import SupportIntegratedMAPPO


def initialize_exploratory_prior(model):
    with torch.no_grad():
        model.pricing.actor.weight.zero_()
        model.pricing.actor.weight[:,0,:]=torch.log(torch.tensor([.19,.19,.19,.19,.24]))
        model.ordering.weight.zero_()
        model.ordering.weight[:,0,:]=torch.log(torch.tensor([.32,.36,.32]))


class ExploratorySampledMAPPO(LinearConditionalCreditMAPPO):
    def __init__(self,env,covers,seed=0):
        super().__init__(env,covers,seed,credit_mode='coordinate');initialize_exploratory_prior(self)
    def checkpoint(self):
        data=super().checkpoint();data['variant']='exploratory_sampled_credit_mappo_v1';return data
    def restore(self,data):
        if data.get('variant')!='exploratory_sampled_credit_mappo_v1':raise ValueError('exploratory sampled schema required')
        super().restore(dict(data,variant='linear_conditional_credit_mappo_v1'))


class ExploratoryIntegratedMAPPO(SupportIntegratedMAPPO):
    def __init__(self,env,covers,seed=0):
        super().__init__(env,covers,seed);initialize_exploratory_prior(self)
    def checkpoint(self):
        data=super().checkpoint();data['variant']='exploratory_integrated_credit_mappo_v1';return data
    def restore(self,data):
        if data.get('variant')!='exploratory_integrated_credit_mappo_v1':raise ValueError('exploratory integrated schema required')
        super().restore(dict(data,variant='support_integrated_mappo_prototype_v1'))
