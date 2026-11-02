"""Seven local features, all five existing physical prices, exact strong greedy prior."""
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from src.algos.structured_pricing_ppo import pricing_features


class FullPriceActor(nn.Module):
    schema = 'local_seven_feature_five_physical_price_actor_v1'
    prior = (.1875, .1875, .1875, .1875, .25)

    def __init__(self, base_prices, phi):
        super().__init__()
        for name, value in (('base_prices', base_prices), ('phi', phi)):
            value = torch.as_tensor(value, dtype=torch.float64)
            if value.shape != (10,) or not torch.isfinite(value).all() or not (value > 0).all():
                raise ValueError('ten fixed positive static provenance values required')
            self.register_buffer(name, value.clone())
        self.register_buffer('temperature', torch.tensor(1.))
        self.weight = nn.Parameter(torch.zeros(10, 7, 5))
        with torch.no_grad():
            self.weight[:, 0, :] = torch.log(torch.tensor(self.prior))

    def set_temperature(self, value):
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not .25 <= value <= 1:
            raise ValueError('finite registered pricing temperature .25..1 required')
        self.temperature.fill_(value)

    def features(self, observation, day):
        return pricing_features(np.asarray(observation, dtype=np.float32), day, self.base_prices.numpy()).astype(np.float32)

    def distribution(self, features):
        if features.shape[-2:] != (10, 7) or not torch.isfinite(features).all():
            raise ValueError('finite seven actor-local features required')
        flat = features.reshape(-1, 10, 7)
        logits = torch.stack([torch.einsum('if,ifk->ik', row, self.weight) for row in flat])
        return Categorical(logits=logits.reshape(features.shape[:-1] + (5,)) / self.temperature)

    def stats(self, features, actions):
        distribution = self.distribution(features)
        return distribution.log_prob(actions).sum(-1), distribution.entropy().sum(-1)

    @torch.no_grad()
    def act(self, features, greedy=False):
        distribution = self.distribution(torch.as_tensor(features))
        action = distribution.probs.argmax(-1) if greedy else distribution.sample()
        return action.numpy(), float(distribution.log_prob(action).sum())


def dual_temperature_transition(after, following, value):
    from outputs.temperature_pilot_replay import strict
    expected = copy.deepcopy(after)
    for role in ('ordering', 'pricing'):
        expected[role + '_temperature'] = value
        expected['model'][role + '.temperature'].fill_(value)
    strict(expected, following)
