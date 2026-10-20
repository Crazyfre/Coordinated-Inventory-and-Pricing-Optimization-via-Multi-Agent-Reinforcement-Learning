import numpy as np
import torch
from src.algos.structured_residual import StructuredResidualActor
from src.algos.local_feature_residual import LocalFeatureResidualActor


def test_local_arithmetic_initial_support_and_gradients():
    torch.set_num_threads(1)
    x=torch.tensor([[0.,0,0,0,0],[0.,0,0,0,100],[10.,15,5,0,10.]])
    torch.manual_seed(11);base=StructuredResidualActor(5,1,2,[6,-1,0],8)
    torch.manual_seed(11);actor=LocalFeatureResidualActor(5,1,2,[6,-1,0],8)
    raw=torch.tensor([[[0.,0,0]],[[20.,0,120]],[[2.,6,6]]])
    torch.testing.assert_close(actor.local_features(x),torch.sign(raw)*torch.log1p(raw.abs()))
    torch.testing.assert_close(actor(x)[0].probs,base(x)[0].probs)
    y=torch.tensor([0,40,8]);loss=-actor(x)[0].log_prob(y).mean();loss.backward()
    assert torch.isfinite(actor.heads[0].weight.grad).all()
    assert actor.heads[0].weight.grad[:,-3:].abs().sum()>0
    assert sum(p.numel() for p in actor.parameters())-sum(p.numel() for p in base.parameters())==15
    for row,support in enumerate([[0,1,2],[38,39,40],[4,5,6,7,8]]):
        np.testing.assert_array_equal(torch.where(actor(x)[0].probs[row]>0)[0],support)
