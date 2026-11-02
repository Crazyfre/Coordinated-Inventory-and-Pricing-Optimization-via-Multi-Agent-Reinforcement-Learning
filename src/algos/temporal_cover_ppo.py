"""Episodic latent-plan PPO; deliberately distinct from daily MAPPO."""
import numpy as np
import torch
from torch import nn


def plan_log_prob(plans,mean,sigma=.75):
    if plans.ndim!=3 or plans.shape[-1]!=10 or mean.shape!=(10,) or sigma<=0:
        raise ValueError('batch by independent draws by ten covers required')
    return (-.5*((plans-mean)/sigma)**2-np.log(sigma*np.sqrt(2*np.pi))).sum((1,2))


def paired_advantages(returns):
    r=np.asarray(returns,float)
    if r.shape!=(8,2) or not np.isfinite(r).all():raise ValueError('eight finite plus/minus return pairs required')
    pair_mean=r.mean(1)
    baseline=np.array([np.mean(np.delete(pair_mean,i)) for i in range(8)])
    a=(r-baseline[:,None]).ravel()
    return ((a-a.mean())/(a.std()+1e-8)).astype(np.float32)


class TemporalCoverPPO:
    def __init__(self,covers,arm,seed=801):
        if arm not in ['episode','daily']:raise ValueError('episode or daily arm required')
        c=np.asarray(covers,np.float32)
        if c.shape!=(10,) or not np.isfinite(c).all():raise ValueError('ten finite covers required')
        self.arm=arm;self.draws=1 if arm=='episode' else 28
        self.mean=nn.Parameter(torch.tensor(c.copy()));self.optimizer=torch.optim.Adam([self.mean],lr=.03)
        self.rng=np.random.default_rng(seed)

    def plans(self):
        epsilon=self.rng.normal(size=(8,self.draws,10))
        center=self.mean.detach().numpy().astype(float)
        plans=np.stack([center+.75*epsilon,center-.75*epsilon],axis=1).reshape(16,self.draws,10).astype(np.float32)
        return epsilon,plans

    def update(self,plans,returns):
        z=torch.as_tensor(plans,dtype=torch.float32)
        old=plan_log_prob(z,self.mean).detach().clone()
        advantage=torch.tensor(paired_advantages(returns));steps=0
        for _ in range(4):
            lp=plan_log_prob(z,self.mean);ratio=(lp-old).exp()
            if float((((ratio-1)-(lp-old)).mean()).detach())>.02:break
            loss=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
            self.optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_([self.mean],.5)
            self.optimizer.step();steps+=1
        with torch.no_grad():
            lp=plan_log_prob(z,self.mean);ratio=(lp-old).exp()
        return dict(old_logp=old.tolist(),advantage=advantage.tolist(),steps=steps,
            kl=float(((ratio-1)-(lp-old)).mean()),loss=float(loss.detach()))

    def state(self):
        return dict(mean=self.mean.detach().clone(),optimizer=self.optimizer.state_dict(),
            rng=self.rng.bit_generator.state,arm=self.arm)


def evaluation_plan(mean,arm,key,greedy=False):
    c=np.asarray(mean,np.float32);draws=1 if arm=='episode' else 28
    if greedy:return np.broadcast_to(c,(draws,10)).copy()
    rng=np.random.default_rng(np.random.SeedSequence([801,key,900]))
    return (c.astype(float)+.75*rng.normal(size=(draws,10))).astype(np.float32)
