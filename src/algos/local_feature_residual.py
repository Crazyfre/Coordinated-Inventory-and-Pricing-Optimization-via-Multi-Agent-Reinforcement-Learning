"""Diagnostic own-SKU features appended to structured residual action heads."""
import torch
from torch import nn
from torch.distributions import Categorical
from src.algos.structured_residual import StructuredResidualActor


class LocalFeatureResidualActor(StructuredResidualActor):
    variant='local_feature_structured_residual_actor_v1'

    def __init__(self,obs_dim,n,pipe_horizon,coefficients,hidden=64):
        super().__init__(obs_dim,n,pipe_horizon,coefficients,hidden)
        self.heads=nn.ModuleList([nn.Linear(hidden+3,5) for _ in range(n)])
        for head in self.heads:
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def local_features(self,obs):
        position=obs[...,:self.n]+obs[...,self.n:self.n*(1+self.pipe_horizon)].reshape(
            *obs.shape[:-1],self.n,self.pipe_horizon).sum(-1)
        forecast=obs[...,-self.n:].double()/5
        position=position.double()/5
        c=self.center_coefficients
        score=c[0]*forecast+c[1]*position+c[2]
        raw=torch.stack([forecast,position,score],-1).to(obs.dtype)
        return torch.sign(raw)*torch.log1p(raw.abs())

    def forward(self,obs):
        center=self.centers(obs);h=self.body(self.input_transform(obs));local=self.local_features(obs)
        result=[]
        for i,head in enumerate(self.heads):
            indices=(center[...,i,None]+torch.arange(-2,3,device=obs.device)).clamp(0,40)
            valid=torch.ones_like(indices,dtype=torch.bool)
            valid[...,1:]=indices[...,1:]!=indices[...,:-1]
            logits=head(torch.cat([h,local[...,i,:]],-1))+torch.where(indices==center[...,i,None],0.,-8.)
            logits=logits.masked_fill(~valid,-torch.inf)
            physical=torch.full((*indices.shape[:-1],41),-torch.inf,device=obs.device)
            result.append(Categorical(logits=physical.scatter_reduce(-1,indices,logits,reduce='amax',include_self=True)))
        return result
