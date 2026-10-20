"""Numerical transforms only; no observation fields or economics removed."""
import torch
from torch import nn


class SignedLog(nn.Module):
    """Invertible stateless compression; avoids tanh saturation by large counts."""
    def forward(self, x):
        # Infinite headroom in A7 is an unbounded-capacity observation sentinel,
        # not a physical capacity cap. Finite values remain exactly invertible.
        finite = torch.nan_to_num(x, nan=float('nan'))
        return torch.sign(finite)*torch.log1p(torch.abs(finite))
