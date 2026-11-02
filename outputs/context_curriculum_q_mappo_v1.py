"""Uninstalled context-table pricing with accepted role-calibrated joint PPO."""
import torch
from outputs.role_calibrated_q_mappo_v1 import RoleCalibratedQMAPPO
from outputs.context_bin_price_actor_v1 import ContextBinPriceActor


class ContextCurriculumQMAPPO(RoleCalibratedQMAPPO):
    variant = 'context_table_curriculum_role_calibrated_q_mappo_v1'
    design = 'local_context_price_fixed_forecast_curriculum_role_q_separate_ppo_v1'
    schemas = dict(RoleCalibratedQMAPPO.schemas,
        context_schema='week4_cover5_score9_shared_action_table_v1',
        curriculum_schema='train_eight_uniform_eight_fixed369_public_forecast_v1')

    def __init__(self, env, covers, phi, seed=0, *, critic_mode='central'):
        super().__init__(env, covers, phi, seed, critic_mode=critic_mode)
        self.pricing = ContextBinPriceActor(env.base_price, phi)
        self.actor_optimizers['pricing'] = torch.optim.Adam(self.pricing.parameters(), lr=.003)

    def restore(self, data):
        # These are protocol constants, never learned checkpoint state.
        state = data.get('model', {})
        golden = {'pricing.log_prior': torch.log(torch.tensor(self.pricing.prior)),
                  'pricing.logit_gain': torch.tensor(8.)}
        for name, expected in golden.items():
            value = state.get(name)
            if not isinstance(value, torch.Tensor) or value.dtype != expected.dtype or not torch.equal(value, expected):
                raise ValueError('Exact fixed context prior and logit gain required')
        return super().restore(data)
