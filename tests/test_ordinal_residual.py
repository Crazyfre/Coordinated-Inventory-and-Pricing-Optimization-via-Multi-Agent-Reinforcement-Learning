import torch
from src.algos.ordinal_residual import OrdinalResidualActor
from src.algos.local_only_residual import LocalOnlyResidualActor


def test_initial_physical_match_and_stable_ordered_probabilities():
    torch.set_num_threads(1);a=OrdinalResidualActor(5,1,2,[6,-1,0],8);b=LocalOnlyResidualActor(5,1,2,[6,-1,0],8)
    x=torch.tensor([[0.,0,0,0,0],[0.,0,0,0,100],[10.,15,5,0,10.]])
    torch.testing.assert_close(a(x)[0].probs,b(x)[0].probs)
    score=torch.tensor([-1000.,-10.,0.,10.,1000.],requires_grad=True)
    logp=a.ordinal_logp(score);assert torch.isfinite(logp).all()
    torch.testing.assert_close(logp.exp().sum(-1),torch.ones(5,dtype=torch.float64))
    assert torch.all(a.thresholds().diff()>0)
    means=(logp.exp()*torch.arange(5)).sum(-1);assert torch.all(means.diff()>0)
    (-logp[:,2].mean()).backward();assert torch.isfinite(score.grad).all() and torch.isfinite(a.raw_gaps.grad).all()


def test_physical_supervision_reaches_score_and_thresholds():
    a=OrdinalResidualActor(5,1,2,[6,-1,0],8)
    x=torch.tensor([[10.,15,5,0,10.]])
    (-a(x)[0].log_prob(torch.tensor([8])).mean()).backward()
    assert a.network[-1].weight.grad.abs().sum()>0
    assert a.first.grad.abs().sum()>0
