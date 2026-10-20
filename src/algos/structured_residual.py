"""Explicit structured-center residual PPO; restricted support, fixed price."""
import torch
from torch import nn
from torch.distributions import Categorical
from src.algos.mappo import Actor,HeterogeneousMAPPO
from src.algos.ordering_learning import FixedMarkupActor
from src.algos.structured_imitation import InventoryPositionLinearPolicy


class StructuredResidualActor(Actor):
    def __init__(self,obs_dim,n,pipe_horizon,coefficients,hidden=64):
        super().__init__(obs_dim,[5]*n,hidden,True)
        self.n,self.pipe_horizon=n,pipe_horizon
        self.register_buffer('center_coefficients',torch.tensor(coefficients,dtype=torch.float64))
        for head in self.heads:
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def centers(self,obs):
        position=obs[...,:self.n]+obs[...,self.n:self.n*(1+self.pipe_horizon)].reshape(
            *obs.shape[:-1],self.n,self.pipe_horizon).sum(-1)
        c=self.center_coefficients
        score=c[0]*(obs[...,-self.n:].double()/5)+c[1]*(position.double()/5)+c[2]
        return score.clamp(min=0).ceil().clamp(max=40).long()

    def forward(self,obs):
        center=self.centers(obs)
        h=self.body(self.input_transform(obs))
        result=[]
        for i,head in enumerate(self.heads):
            # Physical duplicates are masked before normalization: one outcome/quantity.
            indices=(center[...,i,None]+torch.arange(-2,3,device=obs.device)).clamp(0,40)
            valid=torch.ones_like(indices,dtype=torch.bool)
            valid[...,1:]=indices[...,1:]!=indices[...,:-1]
            is_center=indices==center[...,i,None]
            logits=(head(h)+torch.where(is_center,0.,-8.)).masked_fill(~valid,-torch.inf)
            physical=torch.full((*indices.shape[:-1],41),-torch.inf,device=obs.device)
            # scatter_reduce keeps the valid duplicate, including clipped endpoints.
            physical=physical.scatter_reduce(-1,indices,logits,reduce='amax',include_self=True)
            result.append(Categorical(logits=physical))
        return result


class StructuredResidualMAPPO(HeterogeneousMAPPO):
    variant='structured_center_residual_ppo_v1'
    def __init__(self,env,center_path):
        if env.pricing_scope!='shared' or list(env.grid)!=list(range(0,201,5)):
            raise ValueError('fixed shared-price grid fixture required')
        center=InventoryPositionLinearPolicy.load(center_path,env.n,env.obs_pipe_horizon)
        super().__init__(env,normalize_inputs=True,target_kl=.02)
        self.pricing.__class__=FixedMarkupActor
        obs,_=env.reset(seed=0)
        self.replenishment=StructuredResidualActor(len(obs['replenishment']),env.n,
            env.obs_pipe_horizon,center.coefficients)
        self.optimizers[1]=torch.optim.Adam(self.replenishment.parameters(),lr=.0003)
        self.spec=dict(n=env.n,pipe_horizon=env.obs_pipe_horizon,
                       coefficients=center.coefficients.tolist(),offsets=[-2,-1,0,1,2],initial_logit_gap=8.)

    def save(self,path):
        super().save(path)
        data=torch.load(path,weights_only=True)
        data.update(variant=self.variant,spec=self.spec,fixed_price_index=4)
        torch.save(data,path)

    def load(self,path):
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data.get('variant')!=self.variant or data.get('spec')!=self.spec or data.get('fixed_price_index')!=4:
            raise ValueError('matching structured residual schema required')
        super().load(path)
