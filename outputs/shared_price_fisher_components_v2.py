"""Exact shared-context price Fisher blocks; uninstalled, no optimization on import."""
import math
import torch

GROUPS=20
WIDTH=36

def _inputs(actor,features):
    weight=actor.weight
    if weight.shape!=(180,4) or weight.dtype not in (torch.float32,torch.float64):
        raise ValueError('Existing180x4 float pricing table required')
    if not torch.isfinite(weight).all() or float(actor.logit_gain)!=8:
        raise ValueError('Finite table and exact existing gain8 required')
    temperature=float(actor.temperature)
    if not math.isfinite(temperature) or not .25<=temperature<=1:
        raise ValueError('Existing temperature .25..1 required')
    if not isinstance(features,torch.Tensor) or features.device!=weight.device or features.dtype not in (torch.float32,torch.float64) or features.ndim!=3 or features.shape[0]<1 or features.shape[1:]!=(10,4):
        raise ValueError('Complete N-by10-by4 feature tensor on actor device required')
    if not torch.isfinite(features).all() or (features!=features.floor()).any() or (features<0).any() or (features>=180).any():
        raise ValueError('Finite integral context indices0..179 required')
    index=features.long();group=index//9
    if not torch.equal(group,group[...,:1].expand_as(group)):
        raise ValueError('All action contexts must share actual week/cover group')
    return index,group[...,0],temperature

@torch.no_grad()
def fisher_blocks(actor,features):
    index,group,temperature=_inputs(actor,features)
    p=actor.distribution(features).probs.detach().to(torch.float64)
    if p.shape!=features.shape[:-1]+(5,) or not torch.isfinite(p).all() or (p<=0).any():
        raise ValueError('Exact existing full-support five-price distribution required')
    if not torch.allclose(p.sum(-1),torch.ones_like(p[...,0]),atol=2e-6,rtol=0):
        raise ValueError('Normalized categorical probabilities required')
    actions=torch.arange(4,device=index.device)
    raw=actor.logit_gain*actor.weight[index,actions]
    # Runtime clamp convention is authoritative at nondifferentiable bounds.
    with torch.enable_grad():
        probe=raw.detach().requires_grad_(True)
        derivative,=torch.autograd.grad(probe.clamp(-6,6).sum(),probe)
    h=derivative.to(torch.float64)*(8/temperature)
    positions=((index%9)*4+actions).reshape(-1,4)
    groups=group.reshape(-1);free=p[...,:4].reshape(-1,4);h=h.reshape(-1,4)
    result=torch.zeros((20,36,36),dtype=torch.float64,device=index.device)
    count=len(groups)
    for a in range(4):
        for b in range(4):
            covariance=free[:,a]*((1. if a==b else 0.)-free[:,b])*h[:,a]*h[:,b]/count
            result.index_put_((groups,positions[:,a],positions[:,b]),covariance,accumulate=True)
    if not torch.isfinite(result).all():raise ValueError('Finite Fisher required')
    return result

def damped_direction(blocks,gradient,*,damping):
    if not isinstance(blocks,torch.Tensor) or blocks.shape!=(20,36,36) or blocks.dtype!=torch.float64:
        raise ValueError('20x36x36 float64 Fisher required')
    if not isinstance(gradient,torch.Tensor) or gradient.shape!=(180,4) or gradient.device!=blocks.device or not torch.isfinite(gradient).all():
        raise ValueError('Complete finite180x4 gradient on Fisher device required')
    if not math.isfinite(float(damping)) or float(damping)<=0 or not torch.isfinite(blocks).all():
        raise ValueError('Finite positive damping and finite Fisher required')
    if not torch.allclose(blocks,blocks.transpose(-1,-2),atol=1e-9,rtol=0):
        raise ValueError('Symmetric Fisher required')
    if float(torch.linalg.eigvalsh(blocks).min()) < -1e-8:
        raise ValueError('Positive semidefinite Fisher required')
    matrix=blocks+float(damping)*torch.eye(36,dtype=torch.float64,device=blocks.device)
    answer=torch.linalg.solve(matrix,gradient.to(torch.float64).reshape(20,36,1)).reshape(180,4)
    if not torch.isfinite(answer).all():raise ValueError('Finite solved direction required')
    return answer

def mean_categorical_kl(old,new):
    if not isinstance(old,torch.Tensor) or not isinstance(new,torch.Tensor) or old.shape!=new.shape or old.ndim!=3 or old.shape[1:]!=(10,5) or old.shape[0]<1 or old.device!=new.device:
        raise ValueError('Matched complete N-by10-by5 distributions required')
    for p in (old,new):
        if not torch.isfinite(p).all() or (p<=0).any() or not torch.allclose(p.sum(-1),torch.ones_like(p[...,0]),atol=2e-6,rtol=0):
            raise ValueError('Finite normalized full-support probabilities required')
    a,b=old.to(torch.float64),new.to(torch.float64)
    return (a*(a.log()-b.log())).sum(-1).mean()
