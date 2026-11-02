"""Uninstalled prior-only experimental factor; no automatic fitting."""
import torch
from outputs.exploratory_credit_mappo import ExploratoryIntegratedMAPPO


class AnchoredPricingIntegratedMAPPO(ExploratoryIntegratedMAPPO):
    def __init__(self, env, covers, seed=0):
        super().__init__(env, covers, seed)
        with torch.no_grad():
            self.pricing.actor.weight.zero_()
            self.pricing.actor.weight[:, 0, :] = torch.log(torch.tensor([.005, .005, .005, .005, .98]))

    def checkpoint(self):
        data = super().checkpoint()
        data['variant'] = 'anchored_pricing_integrated_mappo_v1'
        data['pricing_initial_prior'] = 'original_concentrated_v1'
        return data

    def restore(self, data):
        if (data.get('variant') != 'anchored_pricing_integrated_mappo_v1'
                or data.get('pricing_initial_prior') != 'original_concentrated_v1'):
            raise ValueError('anchored-pricing integrated schema required')
        super().restore(dict(data, variant='exploratory_integrated_credit_mappo_v1'))
