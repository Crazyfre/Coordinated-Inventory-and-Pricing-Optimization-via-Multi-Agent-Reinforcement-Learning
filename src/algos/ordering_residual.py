"""Explicit local-support forecast-relative policy, NOT vanilla MAPPO.

Residual offsets -2,-1,0,1,2 grid indices (quantity -10..10) around cover6.
Physical grid stays 0..200 by5. Clipped duplicates have summed probability.
Only local support is executable; retained 41-head tensors allow paired RNG
initialization but only their first five outputs parameterize residuals.
"""
import torch
from torch.distributions import Categorical
from src.algos.forecast_prior import ForecastPriorActor


class ForecastRelativeResidualActor(ForecastPriorActor):
    def forward(self, obs):
        pos = obs[..., :self.n] + obs[..., self.n:self.n*(1+self.pipe_horizon)].reshape(*obs.shape[:-1],self.n,self.pipe_horizon).sum(-1)
        center = ((self.cover_days*obs[..., -self.n:]-pos).clamp(min=0)/self.order_step).ceil().clamp(max=40).long()
        offsets = torch.tensor([-2,-1,0,1,2],device=obs.device)
        bias = torch.tensor([-1.,-.5,0.,-.5,-1.],device=obs.device)
        h = self.body(self.input_transform(obs))
        distributions = []
        for i,head in enumerate(self.heads):
            local = (head(h)[...,:5]+bias).softmax(-1)
            indices = (center[...,i,None]+offsets).clamp(0,40)
            physical = torch.zeros(*local.shape[:-1],41,device=obs.device,dtype=obs.dtype).scatter_add(-1,indices,local)
            distributions.append(Categorical(probs=physical))
        return distributions
