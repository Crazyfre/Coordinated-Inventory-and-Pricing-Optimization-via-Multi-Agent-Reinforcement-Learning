import numpy as np
import torch
from scripts.diagnose_residual_imitation import ProjectedPolicy, physical_nll
from scripts.diagnose_cover_search import CoverPolicy
from src.algos.structured_imitation import InventoryPositionLinearPolicy
from src.algos.structured_residual import StructuredResidualActor


def test_projected_labels_have_finite_trainable_physical_likelihood():
    torch.set_num_threads(1); torch.manual_seed(11)
    actor=StructuredResidualActor(5,1,2,[6,-1,0],8)
    x=torch.tensor([[0.,0,0,0,0],[0.,0,0,0,100],[10.,15,5,0,10.]])
    policy=ProjectedPolicy(InventoryPositionLinearPolicy(1,2,[6,-1,0]),CoverPolicy(1,2,7.5))
    y=torch.tensor(np.asarray([policy.act(row.numpy())[0] for row in x]))
    assert y.flatten().tolist()==[0,40,8]
    before=physical_nll(actor,x,y).item()
    optimizer=torch.optim.Adam(actor.parameters(),lr=.01)
    for _ in range(20):
        loss=physical_nll(actor,x,y);optimizer.zero_grad();loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)
        optimizer.step()
    assert physical_nll(actor,x,y).item()<before
    for row,d in enumerate(actor(x)[0].probs):
        center=actor.centers(x)[row,0].item()
        assert torch.where(d>0)[0].tolist()==list(range(max(0,center-2),min(40,center+2)+1))
