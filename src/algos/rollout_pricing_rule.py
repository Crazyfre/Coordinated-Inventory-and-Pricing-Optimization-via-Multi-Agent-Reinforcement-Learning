"""Compact rollout-fitted pricing initialization; not PPO or MAPPO."""
import numpy as np

THRESHOLDS = (None, 3., 2.5, 2., 1.5, 1., .5, 0.)


def mask(obs, threshold):
    x=np.asarray(obs,np.float32)
    if x.ndim!=1 or x.size<180 or not np.isfinite(x).all():
        raise ValueError('finite pricing observation with stock and final forecast/headroom blocks required')
    if threshold is None:return np.zeros(10,dtype=bool)
    if threshold not in THRESHOLDS:raise ValueError('predeclared threshold required')
    return x[:10] > float(threshold)*7*x[-20:-10]


def act(obs, thresholds):
    if len(thresholds)!=10:raise ValueError('one threshold per SKU required')
    return np.array([3 if mask(obs,t)[i] else 4 for i,t in enumerate(thresholds)],dtype=int)


def fit(rows):
    if [(r['episode'],r['day']) for r in rows]!=[(ep,d) for ep in range(16) for d in [1,8,15,22]]:
        raise ValueError('complete ordered 16-window weekly fit grid required')
    advantages=np.asarray([r['advantages'] for r in rows],float)
    if advantages.shape!=(64,16,10) or not np.isfinite(advantages).all():
        raise ValueError('16 paired future advantages for each state and SKU required')
    values=advantages.mean(1)
    masks=np.stack([[mask(r['obs'],t) for t in THRESHOLDS] for r in rows])
    scores=(masks*values[:,None,:]).mean(0).T
    counts=masks.sum(0).T
    selected=np.argmax(scores,axis=1)
    return dict(thresholds=[THRESHOLDS[i] for i in selected],scores=scores.tolist(),
        discount_state_counts=counts.tolist(),selected_indices=selected.tolist(),
        method='rollout-fitted threshold initialization; not PPO or MAPPO')
