"""Explicit imitation-assisted ordering; never vanilla MAPPO.

Supervised initialization uses only train-window teacher states. Residual phase
restricts support to physical quantities within +/-10 units of the teacher.
"""
import torch
from torch.distributions import Categorical
from src.algos.mappo import Actor, HeterogeneousMAPPO
from src.algos.ordering_learning import FixedMarkupActor


def teacher_indices(obs, n, pipe_horizon, order_step=5, cover=6):
    position = obs[..., :n] + obs[..., n:n*(1+pipe_horizon)].reshape(
        *obs.shape[:-1], n, pipe_horizon).sum(-1)
    return ((cover*obs[..., -n:] - position).clamp(min=0)/order_step).ceil().clamp(max=40).long()


class TeacherResidualActor(Actor):
    """Condition learned full-grid logits on five physical teacher offsets."""
    def forward(self, obs):
        center = teacher_indices(obs, self.n, self.pipe_horizon, self.order_step)
        offsets = torch.tensor([-2, -1, 0, 1, 2], device=obs.device)
        h = self.body(self.input_transform(obs))
        distributions = []
        for i, head in enumerate(self.heads):
            indices = (center[..., i, None] + offsets).clamp(0, 40)
            # Duplicated boundary quantities get one logit/probability, not extra mass.
            valid = torch.ones_like(indices, dtype=torch.bool)
            valid[..., 1:] = indices[..., 1:] != indices[..., :-1]
            logits = head(h).gather(-1, indices).masked_fill(~valid, -torch.inf)
            local = logits.softmax(-1)
            physical = torch.zeros(*local.shape[:-1], 41, device=obs.device,
                                   dtype=obs.dtype).scatter_add(-1, indices, local)
            distributions.append(Categorical(probs=physical))
        return distributions


class ImitationAssistedMAPPO(HeterogeneousMAPPO):
    """Fixed shared markup fixture with explicit full-grid/residual variants."""
    def __init__(self, env, *, residual=False, **kwargs):
        if getattr(env, 'pricing_scope', 'shared') != 'shared':
            raise ValueError('ordering fixture requires explicit shared fixed pricing')
        if (len(env.grid) != 41 or env.config.replenishment.order_step != 5
                or not all(int(v) == 41 for v in env.action_space.nvec)):
            raise ValueError('imitation ordering requires physical grid 0..200 by5')
        super().__init__(env, **kwargs)
        self.pricing.__class__ = FixedMarkupActor
        self.residual = bool(residual)
        self.ordering_spec = dict(n=env.n, pipe_horizon=env.obs_pipe_horizon,
                                 order_step=env.config.replenishment.order_step)
        if self.residual:
            self.replenishment.__class__ = TeacherResidualActor
            for name, value in self.ordering_spec.items():
                setattr(self.replenishment, name, value)

    def save(self, path):
        super().save(path)
        data = torch.load(path, map_location='cpu', weights_only=True)
        data.update(variant='imitation_assisted_ordering_v1', residual=self.residual,
                    ordering_spec=self.ordering_spec, fixed_price_index=4)
        torch.save(data, path)

    def load(self, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if (data.get('variant') != 'imitation_assisted_ordering_v1'
                or data.get('residual') != self.residual
                or data.get('ordering_spec') != self.ordering_spec
                or data.get('fixed_price_index') != 4):
            raise ValueError('matching explicit imitation ordering checkpoint required')
        super().load(path)
