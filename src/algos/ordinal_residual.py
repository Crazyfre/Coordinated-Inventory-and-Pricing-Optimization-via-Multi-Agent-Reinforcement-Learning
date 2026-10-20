"""Ordered-logistic change in residual log probabilities; original physical map."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from src.algos.local_only_residual import LocalOnlyResidualActor
from src.algos.ppo import orthogonal_init


class OrdinalResidualActor(LocalOnlyResidualActor):
    variant='ordinal_local_residual_actor_v1'

    def __init__(self,obs_dim,n,pipe_horizon,coefficients,hidden=64):
        nn.Module.__init__(self);self.n,self.pipe_horizon=n,pipe_horizon
        self.register_buffer('center_coefficients',torch.tensor(coefficients,dtype=torch.float64))
        self.network=nn.Sequential(nn.Linear(2,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.network.apply(orthogonal_init);nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        q=torch.softmax(torch.tensor([-8.,-8.,0.,-8.,-8.],dtype=torch.float64),0)
        thresholds=torch.logit(q.cumsum(0)[:-1])
        self.first=nn.Parameter(thresholds[:1].clone())
        gaps=thresholds.diff()-1e-6
        self.raw_gaps=nn.Parameter(torch.log(torch.expm1(gaps)))
        self.register_buffer('initial_logp',self.ordinal_logp(torch.zeros(1,dtype=torch.float64)).detach()[0])

    def thresholds(self):
        return torch.cat([self.first,self.first+torch.cumsum(F.softplus(self.raw_gaps)+1e-6,0)])

    def ordinal_logp(self,score):
        t=self.thresholds();z=t-score[...,None].double()
        middle=F.logsigmoid(z[...,1:])+F.logsigmoid(-z[...,:-1])+torch.log(-torch.expm1(t[:-1]-t[1:]))
        return torch.cat([F.logsigmoid(z[...,:1]),middle,F.logsigmoid(-z[...,-1:])],-1)

    def forward(self,obs):
        center=self.centers(obs);score=self.network(self.local_inputs(obs)).squeeze(-1)
        change=(self.ordinal_logp(score)-self.initial_logp).to(obs.dtype);result=[]
        for i in range(self.n):
            indices=(center[...,i,None]+torch.arange(-2,3,device=obs.device)).clamp(0,40)
            valid=torch.ones_like(indices,dtype=torch.bool);valid[...,1:]=indices[...,1:]!=indices[...,:-1]
            logits=(change[...,i,:]+torch.where(indices==center[...,i,None],0.,-8.)).masked_fill(~valid,-torch.inf)
            physical=torch.full((*indices.shape[:-1],41),-torch.inf,device=obs.device)
            result.append(Categorical(logits=physical.scatter_reduce(-1,indices,logits,reduce='amax',include_self=True)))
        return result
