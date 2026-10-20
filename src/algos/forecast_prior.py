"""Opt-in informed MAPPO variant, NOT vanilla PPO or a default replacement.

Adds a fixed forecast/base-stock categorical log-prior to learned residual
logits. Uses only the actor's local raw observation (including staged price
forecast); no env access during forward/update and no reward/physics changes.
"""
import math
import torch
from torch.distributions import Categorical
from src.algos.mappo import Actor, HeterogeneousMAPPO


class ForecastPriorMAPPO(HeterogeneousMAPPO):
    """Explicit opt-in subclass; vanilla constructor/checkpoints stay unchanged."""
    def __init__(self, env, hidden=64, lr=3e-4, **kwargs):
        if len(env.grid)!=41 or not all(int(x)==41 for x in env.action_space.nvec):
            raise ValueError('informed variant requires the unchanged 41-action grid')
        super().__init__(env,hidden=hidden,lr=lr,**kwargs)
        self.prior_settings = dict(n=env.n,pipe_horizon=env.obs_pipe_horizon,
                                   order_step=env.config.replenishment.order_step,
                                   cover_days=6.,width=2.,uniform_mass=.05)
        # Reuse identical initialized parameters and optimizer bindings. Only
        # the distribution computation changes, not parameter count or RNG.
        self.replenishment.__class__ = ForecastPriorActor
        for name,value in self.prior_settings.items():
            setattr(self.replenishment,name,value)

    def save(self, path):
        super().save(path)
        data = torch.load(path,map_location='cpu',weights_only=True)
        data.update(variant='forecast_prior_mappo',prior_settings=self.prior_settings)
        torch.save(data,path)

    def load(self, path):
        data = torch.load(path,map_location='cpu',weights_only=True)
        if data.get('variant') != 'forecast_prior_mappo' or data.get('prior_settings') != self.prior_settings:
            raise ValueError('requires matching explicitly labeled forecast-prior checkpoint')
        super().load(path)


class ForecastPriorActor(Actor):
    def __init__(self, obs_dim, nvec, hidden, *, n, pipe_horizon, order_step,
                 normalize_inputs=False, cover_days=6., width=2., uniform_mass=.05):
        if any(int(x) != 41 for x in nvec) or len(nvec) != n:
            raise ValueError('informed variant requires the unchanged 41-action grid')
        if not all(math.isfinite(x) and x>0 for x in (order_step,cover_days,width)):
            raise ValueError('positive finite prior settings required')
        if not 0 < uniform_mass < 1:
            raise ValueError('uniform_mass must be strictly between zero and one')
        super().__init__(obs_dim,nvec,hidden,normalize_inputs)
        self.n, self.pipe_horizon, self.order_step = n, pipe_horizon, order_step
        self.cover_days, self.width, self.uniform_mass = cover_days,width,uniform_mass

    def forward(self, obs):
        position = obs[..., :self.n] + obs[..., self.n:self.n*(1+self.pipe_horizon)].reshape(
            *obs.shape[:-1], self.n,self.pipe_horizon).sum(-1)
        forecast = obs[..., -self.n:]
        target = ((self.cover_days*forecast-position).clamp(min=0)/self.order_step).ceil().clamp(max=40)
        indices = torch.arange(41,device=obs.device,dtype=obs.dtype)
        base = (-(indices-target.unsqueeze(-1)).abs()/self.width).softmax(-1)
        prior = ((1-self.uniform_mass)*base+self.uniform_mass/41).log()
        h = self.body(self.input_transform(obs))
        return [Categorical(logits=head(h)+prior[...,i,:]) for i,head in enumerate(self.heads)]
