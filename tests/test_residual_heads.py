import torch
from torch.distributions import Categorical
from scripts.diagnose_residual_heads import masked_head_loss


def test_head_loss_uses_selected_rows_and_equal_head_weight():
    logits=[torch.tensor([[2.,0.],[0.,1.],[1.,0.]],requires_grad=True) for _ in range(2)]
    ds=[Categorical(logits=v) for v in logits];y=torch.tensor([[0,1],[1,0],[1,0]])
    loss=masked_head_loss(ds,y,{0:torch.tensor([0,1]),1:torch.tensor([2])})
    expected=(-ds[0].log_prob(y[:,0])[:2].mean()-ds[1].log_prob(y[:,1])[2])/2
    torch.testing.assert_close(loss,expected);loss.backward()
    assert torch.equal(logits[0].grad[2],torch.zeros(2))
    assert torch.equal(logits[1].grad[:2],torch.zeros(2,2))
    assert logits[0].grad[:2].abs().sum()>0 and logits[1].grad[2].abs().sum()>0
