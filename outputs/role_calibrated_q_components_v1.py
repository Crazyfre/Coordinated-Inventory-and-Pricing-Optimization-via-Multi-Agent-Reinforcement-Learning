"""Uninstalled local-economic actor and calibrated anchored-Q building blocks."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from outputs.full_price_actor_v1 import FullPriceActor
from outputs.economic_proxy_rule_v2 import scores
from src.algos.structured_pricing_ppo import pricing_features


ROLES = ('pricing', 'ordering')


def role_index(role):
    if role not in ROLES:
        raise ValueError('Known pricing/ordering role required')
    return ROLES.index(role)


class EconomicFeaturePriceActor(FullPriceActor):
    schema = 'local_seven_plus_four_economic_score_features_v1'

    def __init__(self, base_prices, phi):
        super().__init__(base_prices, phi)
        self.weight = nn.Parameter(torch.zeros(10, 11, 5))
        with torch.no_grad():
            self.weight[:, 0, :] = torch.log(torch.tensor(self.prior))

    def features(self, observation, day):
        x = np.asarray(observation, dtype=np.float32)
        base = self.base_prices.numpy()
        original = pricing_features(x, day, base)
        proxy = scores(x, day, base, self.phi.numpy())
        extra = np.clip((proxy[:, :4] - proxy[:, 4:5]) / 100, -3, 3)
        return np.concatenate((original, extra), axis=-1).astype(np.float32)

    def distribution(self, features):
        if features.shape[-2:] != (10, 11) or not torch.isfinite(features).all():
            raise ValueError('Finite11feature actor-local inputs required')
        flat = features.reshape(-1, 10, 11)
        logits = torch.stack([torch.einsum('if,ifk->ik', row, self.weight) for row in flat])
        return Categorical(logits=logits.reshape(features.shape[:-1] + (5,)) / self.temperature)


class LockedRoleRMS(nn.Module):
    """First-batch-only role scales; deliberately no moving mean or later drift."""
    def __init__(self):
        super().__init__()
        self.register_buffer('scales', torch.ones(2, dtype=torch.float64))
        self.register_buffer('locked', torch.zeros(2, dtype=torch.bool))

    @torch.no_grad()
    def lock(self, role, target, mass):
        index = role_index(role)
        if self.locked[index]:
            raise ValueError('Role scale is immutable after first TRAIN batch')
        y, p = target.double(), mass.double()
        if y.ndim != 2 or y.shape != p.shape or not torch.isfinite(y).all() or not torch.isfinite(p).all():
            raise ValueError('Finite root/action targets and probabilities required')
        if (p < 0).any() or not torch.allclose(p.sum(-1), torch.ones(y.shape[0], dtype=torch.float64), atol=4e-6, rtol=0):
            raise ValueError('Complete probability mass at every role root required')
        self.scales[index] = torch.sqrt((p * y.square()).sum() / y.shape[0]).clamp_min(.01)
        self.locked[index] = True

    def scale(self, role):
        return self.scales[role_index(role)].float()


class LaggedQWeight(nn.Module):
    """Observe pre-batch predictions AFTER actor updates; use only in next batch."""
    def __init__(self):
        super().__init__()
        self.register_buffer('qy', torch.zeros(2, dtype=torch.float64))
        self.register_buffer('qq', torch.zeros(2, dtype=torch.float64))
        self.register_buffer('observations', torch.zeros(2, dtype=torch.long))

    @torch.no_grad()
    def coefficient(self, role):
        i = role_index(role)
        return torch.where(self.qq[i] > 1e-12, self.qy[i] / self.qq[i].clamp_min(1e-12),
                           torch.zeros_like(self.qy[i])).clamp(0, 1).float()

    @torch.no_grad()
    def observe(self, role, pre_q, target, mass):
        i = role_index(role)
        q, y, p = pre_q.double(), target.double(), mass.double()
        if q.ndim != 2 or q.shape != y.shape or y.shape != p.shape:
            raise ValueError('Complete selected-root action arrays required')
        if not all(torch.isfinite(v).all() for v in (q, y, p)) or (p < 0).any():
            raise ValueError('Finite targets/predictions and nonnegative masses required')
        if not torch.allclose(p.sum(-1), torch.ones(q.shape[0], dtype=torch.float64), atol=4e-6, rtol=0):
            raise ValueError('Complete action probability masses required')
        self.qy[i].mul_(.8).add_((p * q * y).sum() / q.shape[0] * .2)
        self.qq[i].mul_(.8).add_((p * q.square()).sum() / q.shape[0] * .2)
        self.observations[i].add_(1)


class SharedRoleActionQ(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(width, 64), nn.Tanh())
        self.heads = nn.ModuleDict({r: nn.Linear(64, 5 if r == 'pricing' else 41) for r in ROLES})
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, state, role):
        role_index(role)
        return self.heads[role](self.trunk(state))


class LocalRoleActionQ(nn.Module):
    def __init__(self, widths):
        super().__init__()
        if set(widths) != set(ROLES):
            raise ValueError('Both local role widths required')
        self.networks = nn.ModuleDict({r: nn.Sequential(nn.Linear(widths[r], 64), nn.Tanh(),
            nn.Linear(64, 5 if r == 'pricing' else 41)) for r in ROLES})
        for network in self.networks.values():
            nn.init.zeros_(network[-1].weight)
            nn.init.zeros_(network[-1].bias)

    def forward(self, state, role):
        role_index(role)
        return self.networks[role](state)


def anchor_differences(prediction, anchors):
    if prediction.ndim != 2 or anchors.shape != prediction.shape[:1] or anchors.dtype != torch.long:
        raise ValueError('Root/action heads and one integer physical anchor per root required')
    if (anchors < 0).any() or (anchors >= prediction.shape[1]).any():
        raise ValueError('Anchor inside physical output support required')
    return prediction - prediction.gather(1, anchors[:, None])


def anchored_weighted_loss(prediction, anchors, target, mass, scale):
    if prediction.shape != target.shape or target.shape != mass.shape:
        raise ValueError('Matching complete physical head/target/mass arrays required')
    if not all(torch.isfinite(v).all() for v in (prediction, target, mass, scale)) or float(scale) <= 0:
        raise ValueError('Finite arrays and positive frozen role scale required')
    if (mass < 0).any() or not torch.allclose(mass.sum(-1), torch.ones_like(mass[:, 0]), atol=4e-6, rtol=0):
        raise ValueError('Complete physical probability masses required')
    if not torch.equal(target.gather(1, anchors[:, None]), torch.zeros_like(target[:, :1])):
        raise ValueError('MC residual at the physical strong action must be zero')
    relative = anchor_differences(prediction, anchors)
    return (mass * (relative - target / scale).square()).sum() / target.shape[0]


def cv_components(all_q, coordinate, selected_mc, coefficient):
    if all_q.ndim != 3 or all_q.shape[1] != 10 or coordinate.shape != all_q.shape[:1]:
        raise ValueError('Ten-coordinate frozen Q tables and one selected coordinate required')
    if selected_mc.shape != (all_q.shape[0], all_q.shape[2]):
        raise ValueError('Complete selected-coordinate MC action labels required')
    if coordinate.dtype != torch.long or (coordinate < 0).any() or (coordinate >= 10).any():
        raise ValueError('Selected SKU coordinate inside fixed ten-SKU support required')
    if not 0 <= float(coefficient) <= 1:
        raise ValueError('Lagged coefficient must lie in[0,1]')
    if not torch.isfinite(all_q).all() or not torch.isfinite(selected_mc).all():
        raise ValueError('Finite frozen Q and MC labels required')
    model = coefficient * all_q
    selected = model[torch.arange(all_q.shape[0]), coordinate]
    return model, selected_mc - selected
