"""Diagnostic SKU-shared local-only MLP, with frozen residual support."""
import torch
from torch import nn
from torch.distributions import Categorical
from src.algos.structured_residual import StructuredResidualActor
from src.algos.ppo import orthogonal_init


class LocalOnlyResidualActor(StructuredResidualActor):
    variant='local_only_shared_residual_actor_v1'

    def __init__(self,obs_dim,n,pipe_horizon,coefficients,hidden=64):
        nn.Module.__init__(self)
        self.n,self.pipe_horizon=n,pipe_horizon
        self.register_buffer('center_coefficients',torch.tensor(coefficients,dtype=torch.float64))
        self.network=nn.Sequential(nn.Linear(2,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,5))
        self.network.apply(orthogonal_init)
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def local_inputs(self,obs):
        position=obs[...,:self.n]+obs[...,self.n:self.n*(1+self.pipe_horizon)].reshape(*obs.shape[:-1],self.n,self.pipe_horizon).sum(-1)
        raw=torch.stack([obs[...,-self.n:]/5,position/5],-1)
        return torch.sign(raw)*torch.log1p(raw.abs())

    def forward(self,obs):
        center=self.centers(obs);raw=self.network(self.local_inputs(obs));result=[]
        for i in range(self.n):
            indices=(center[...,i,None]+torch.arange(-2,3,device=obs.device)).clamp(0,40)
            valid=torch.ones_like(indices,dtype=torch.bool);valid[...,1:]=indices[...,1:]!=indices[...,:-1]
            logits=(raw[...,i,:]+torch.where(indices==center[...,i,None],0.,-8.)).masked_fill(~valid,-torch.inf)
            physical=torch.full((*indices.shape[:-1],41),-torch.inf,device=obs.device)
            result.append(Categorical(logits=physical.scatter_reduce(-1,indices,logits,reduce='amax',include_self=True)))
        return result
