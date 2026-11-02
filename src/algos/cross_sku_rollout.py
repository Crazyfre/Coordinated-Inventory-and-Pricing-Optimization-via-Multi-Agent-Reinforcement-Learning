"""Matched observable rollout regressors; neither PPO nor MAPPO."""
import numpy as np
from src.algos.structured_pricing_ppo import pricing_features
from src.algos.cross_sku_pricing_features import features as cross_features


def matrix(obs,day,base_prices,arm):
    if arm not in ['local','cross']:raise ValueError('local or cross arm required')
    local=pricing_features(obs,day,base_prices)[:,1:].astype(float)
    columns=[np.eye(10),local]
    if arm=='cross':columns.append(cross_features(obs,base_prices))
    return np.concatenate(columns,axis=1)


def selection(observations,episode,base_prices):
    if set(observations)!={1,8,15,22}:raise ValueError('four weekly observations required')
    candidates=[(d,i) for d in [1,8,15,22] for i in range(10)]
    score=np.concatenate([cross_features(observations[d],base_prices)[:,3] for d in [1,8,15,22]])
    best=int(np.argmax(score));remaining=[i for i in range(40) if i!=best]
    rng=np.random.default_rng(np.random.SeedSequence([2540000,episode,991]))
    other=remaining[int(rng.integers(39))]
    return [candidates[best],candidates[other]]


def fit(raw,targets,arm):
    if arm not in ['local','cross']:raise ValueError('local or cross arm required')
    x=np.asarray(raw,float);y=np.asarray(targets,float);width=16 if arm=='local' else 20
    if x.shape!=(256,width) or y.shape!=(256,) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('complete finite256-row matrix/targets required')
    mean=np.zeros(width);scale=np.ones(width)
    mean[10:]=x[:,10:].mean(0);scale[10:]=x[:,10:].std(0)
    scale[scale<1e-12]=1
    z=(x-mean)/scale
    w=np.linalg.solve(z.T@z+np.eye(width),z.T@y)
    return dict(arm=arm,mean=mean.tolist(),scale=scale.tolist(),weights=w.tolist())


def act(obs,day,base_prices,model):
    x=matrix(obs,day,base_prices,model['arm'])
    mean,scale,w=[np.asarray(model[k],float) for k in ['mean','scale','weights']]
    if any(a.shape!=(x.shape[1],) or not np.isfinite(a).all() for a in [mean,scale,w]) or np.any(scale<=0):
        raise ValueError('finite matching model arrays and positive scales required')
    gain=((x-mean)/scale@w)*np.asarray(base_prices)
    result=np.full(10,4,dtype=int);best=int(np.argmax(gain))
    if gain[best]>0:result[best]=0
    return result
