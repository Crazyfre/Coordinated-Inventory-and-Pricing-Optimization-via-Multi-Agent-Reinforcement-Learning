"""Uninstalled whole-team PPO credit with matched concentrated-price/soft-order priors."""
import torch
from outputs.linear_conditional_credit_mappo import LinearConditionalCreditMAPPO


class ConcentratedWholeTeamMAPPO(LinearConditionalCreditMAPPO):
    def __init__(self,env,covers,seed=0):
        super().__init__(env,covers,seed,credit_mode='whole')
        with torch.no_grad():
            self.pricing.actor.weight.zero_()
            self.pricing.actor.weight[:,0,:]=torch.log(torch.tensor([.005,.005,.005,.005,.98]))
            self.ordering.weight.zero_()
            self.ordering.weight[:,0,:]=torch.log(torch.tensor([.32,.36,.32]))

    def checkpoint(self):
        data=super().checkpoint();data['variant']='concentrated_whole_team_mappo_v1'
        data['initial_prior_packet']='original_price_exploratory_order_v1'
        return data

    def restore(self,data):
        if (data.get('variant')!='concentrated_whole_team_mappo_v1'
                or data.get('initial_prior_packet')!='original_price_exploratory_order_v1'):
            raise ValueError('concentrated whole-team schema required')
        super().restore(dict(data,variant='linear_conditional_credit_mappo_v1'))
