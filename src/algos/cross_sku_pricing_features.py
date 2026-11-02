"""Fixed actor-observable cross-SKU imbalance features; no policy decisions."""
import numpy as np

NAMES=['other_deficit','other_value_deficit','other_excess','congested_excess_deficit']


def features(obs,base_prices):
    x=np.asarray(obs,dtype=float);b=np.asarray(base_prices,dtype=float)
    if x.ndim!=1 or len(x)<180 or b.shape!=(10,) or not np.isfinite(x).all() or not np.isfinite(b).all() or np.any(b<=0):
        raise ValueError('finite pricing observation and ten positive base prices required')
    stock=x[:10];forecast=x[-20:-10];headroom=x[-10:]
    if np.any(stock<0) or np.any(forecast<0):raise ValueError('nonnegative stock and forecast required')
    deficit=np.maximum(7*forecast-stock,0);excess=np.maximum(stock-7*forecast,0)
    other=(deficit.sum()-deficit)/500
    value=b*deficit
    return np.column_stack([other,(value.sum()-value)/(500*b.mean()),
        (excess.sum()-excess)/500,(excess/500)*other*(1-np.clip(headroom/500,0,1))])
