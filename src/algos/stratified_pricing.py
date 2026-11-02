"""Observable-only rollout regression initialization; not PPO/MAPPO."""
import numpy as np
from src.algos.structured_pricing_ppo import pricing_features


def select_candidates(observations,episode):
    if set(observations)!={1,8,15,22}:raise ValueError('four weekly observations required')
    candidates=[(day,sku) for day in [1,8,15,22] for sku in range(10)]
    ratios=[]
    for day,sku in candidates:
        x=np.asarray(observations[day],np.float32)
        if x.ndim!=1 or x.size<180 or not np.isfinite(x).all():raise ValueError('finite pricing observation required')
        ratios.append(float(x[sku]/(np.float32(1)+np.float32(7)*x[-20+sku])))
    best=int(np.argmax(ratios));remaining=[i for i in range(40) if i!=best]
    rng=np.random.default_rng(np.random.SeedSequence([2500000,episode,991]))
    other=remaining[int(rng.integers(39))]
    return [candidates[best],candidates[other]]


def features(obs,day,base_prices):
    f=pricing_features(obs,day,base_prices)[:,1:].astype(np.float64)
    products=np.stack([f[:,i]*f[:,j] for i in range(6) for j in range(i,6)],axis=1)
    return np.concatenate([np.eye(10),f,products],axis=1)


def fit(matrix,targets):
    x=np.asarray(matrix,np.float64);y=np.asarray(targets,np.float64)
    if x.shape!=(256,37) or y.shape!=(256,) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('complete finite 256-state regression data required')
    return np.linalg.solve(x.T@x+np.eye(37),x.T@y)


def act(obs,day,base_prices,weights):
    w=np.asarray(weights,np.float64);base=np.asarray(base_prices,np.float64)
    if w.shape!=(37,) or not np.isfinite(w).all():raise ValueError('37 finite coefficients required')
    gain=(features(obs,day,base)@w)*base
    result=np.full(10,4,dtype=int);best=int(np.argmax(gain))
    if gain[best]>0:result[best]=3
    return result
