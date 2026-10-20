import torch


def test_residual_maps_and_merges_boundary_probability():
    from src.algos.ordering_residual import ForecastRelativeResidualActor
    actor = ForecastRelativeResidualActor(10, [41], 8, n=1, pipe_horizon=7, order_step=5)
    obs = torch.zeros(1,10)
    dist = actor(obs)[0]
    assert dist.probs.shape == (1,41)
    assert torch.count_nonzero(dist.probs) == 3
    assert dist.probs[0,0] > dist.probs[0,1]
    lp, ent = actor.stats(obs,torch.zeros(1,1,dtype=torch.long))
    assert torch.allclose(lp,dist.log_prob(torch.zeros(1,dtype=torch.long)))
    (-lp.mean()).backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in actor.parameters())
