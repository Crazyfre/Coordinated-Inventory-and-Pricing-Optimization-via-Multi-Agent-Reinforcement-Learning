import torch
from src.algos.mappo import Actor


def test_fixed_price_actor_has_exact_support_and_zero_gradient():
    from src.algos.ordering_learning import FixedMarkupActor
    actor = Actor(3, [5], 8)
    actor.__class__ = FixedMarkupActor
    obs = torch.ones(4, 3)
    dist = actor(obs)[0]
    assert torch.equal(dist.probs, torch.tensor([[0., 0., 0., 0., 1.]]).repeat(4, 1))
    lp, entropy = actor.stats(obs, torch.full((4, 1), 4))
    (-lp.mean() - entropy.mean()).backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in actor.parameters())
    assert actor.act([1., 1., 1.])[0].tolist() == [4]
