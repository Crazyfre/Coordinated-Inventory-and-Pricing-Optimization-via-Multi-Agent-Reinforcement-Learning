"""Unexecuted-estimator primitives. No environment/training runner imports this.

One selected coordinate per on-policy role decision; centralized residual-value
baselines and full remaining-horizon labels still need a separate validated
collector. These primitives alone are not a MAPPO implementation or result.
"""
import torch


def selected_log_prob(distribution, actions, coordinates):
    """Extract one of ten independent physical categorical component scores."""
    values = distribution.log_prob(actions.long())
    if values.ndim != 2 or values.shape[1] != 10:
        raise ValueError('batch of ten independent SKU components required')
    if coordinates.shape != (len(values),) or coordinates.dtype != torch.long:
        raise ValueError('one integer SKU coordinate per decision required')
    if not ((coordinates >= 0) & (coordinates < 10)).all():
        raise ValueError('SKU coordinate outside0..9')
    return values.gather(1, coordinates[:, None]).squeeze(1)


def clipped_terms(new_logp, old_logp, advantages, clip=.1):
    """Per-selected-coordinate PPO terms including uniform ten-SKU weight.

Detached old logps/advantages are required. The caller owns state/coordinate
sampling weights, normalization and full-role KL checks. No normalized/clipped
finite-step unbiasedness claim follows from the score identity at old weights.
"""
    if not (new_logp.shape == old_logp.shape == advantages.shape):
        raise ValueError('matched selected-coordinate scalar shapes required')
    if old_logp.requires_grad or advantages.requires_grad:
        raise ValueError('old policy and return/baseline targets must be detached')
    if not all(torch.isfinite(v).all() for v in (new_logp, old_logp, advantages)):
        raise ValueError('finite log probabilities and advantages required')
    ratio = (new_logp-old_logp).exp()
    return 10*torch.minimum(ratio*advantages, ratio.clamp(1-clip, 1+clip)*advantages)
