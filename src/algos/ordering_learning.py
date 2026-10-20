"""Opt-in fixed-markup diagnostic actor; never a production default."""
import torch
from torch.distributions import Categorical
from src.algos.mappo import Actor


class FixedMarkupActor(Actor):
    """Exactly index4 (1.20x); retains zero-connected autograd for shared loop."""
    def forward(self, obs):
        logits = self.heads[0](self.body(self.input_transform(obs))) * 0
        offset = torch.full_like(logits, -1e9)
        offset[..., 4] = 0
        return [Categorical(logits=logits + offset)]
