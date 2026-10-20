import torch
from src.algos.local_only_residual import LocalOnlyResidualActor
from src.algos.structured_residual import StructuredResidualActor


def test_locality_initial_support_and_gradients():
    torch.set_num_threads(1);a=LocalOnlyResidualActor(6,1,2,[6,-1,0],8)
    b=StructuredResidualActor(6,1,2,[6,-1,0],8)
    x=torch.tensor([[0.,0,0,15,99,0],[0.,0,0,1,2,100],[10.,15,5,1,2,10.]])
    torch.testing.assert_close(a(x)[0].probs,b(x)[0].probs)
    changed=x.clone();changed[:,3:5]=10000
    torch.testing.assert_close(a.local_inputs(x),a.local_inputs(changed))
    torch.testing.assert_close(a(x)[0].probs,a(changed)[0].probs)
    (-a(x)[0].log_prob(torch.tensor([0,40,8])).mean()).backward()
    assert a.network[-1].weight.grad.abs().sum()>0


def test_shared_sku_mapping_equivariance():
    a=LocalOnlyResidualActor(8,2,1,[6,-1,0],8)
    x=torch.tensor([[10.,20.,5.,0.,99.,66.,4.,12.]])
    swapped=x[:,[1,0,3,2,4,5,7,6]]
    for first,second in zip(a(x),reversed(a(swapped))):torch.testing.assert_close(first.probs,second.probs)
