"""Actor-observable arrival context for a separately labeled reward estimator."""
import numpy as np
from src.algos.rollout_correction import features as legacy_features


def arrival_features(obs,n,horizon,sku,delta,day):
    if not 1<=day<=28 or not 0<=sku<n:
        raise ValueError('fixed 28-day fixture and valid SKU required')
    x=np.asarray(obs,dtype=np.float32)
    expected=n*(1+horizon+14+4)
    if x.shape!=(expected,) or not np.isfinite(x).all():
        raise ValueError('finite canonical observation with history14 required')
    stock=x[:n].astype(float)
    pipe=x[n:n+n*horizon].reshape(n,horizon).astype(float)
    forecast=x[-n:].astype(float);price=x[-4*n:-3*n].astype(float)
    log=lambda a:np.sign(a)*np.log1p(np.abs(a))
    per_sku=np.column_stack([log(stock/5),log(pipe/5),log(forecast/5),log(price)])
    context=np.r_[1.,per_sku.ravel(),np.eye(n)[sku],per_sku[sku],float(x[-2*n+sku])/500,(29-day)/28]
    vector=np.r_[delta*context,delta*delta*context]
    # Minimum lead2 plus no terminal charges => no economic action effect.
    return np.zeros_like(vector) if day>=27 else vector


def controlled_legacy_features(obs,n,horizon,sku,delta,day):
    value=legacy_features(obs,n,horizon,sku,delta,day)
    return np.zeros_like(value) if day>=27 else value
