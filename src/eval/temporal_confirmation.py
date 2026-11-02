"""Fixed five-seed crossed confirmation analysis; no model selection."""
import numpy as np

SEEDS=(811,822,833,844,855)
KEYS=tuple(range(2420000,2420128))


def analyze(rows):
    labels=['baseline']+[f'{arm}_{s}' for arm in ['episode','daily'] for s in SEEDS]
    index={(r['label'],r['scenario']):r for r in rows}
    if len(index)!=len(rows) or set(index)!={(l,k) for l in labels for k in KEYS}:
        raise ValueError('complete unique five-seed by128scenario confirmation grid required')
    rng=np.random.default_rng(20261031)
    seed_draws=rng.integers(5,size=(10000,5));key_draws=rng.integers(128,size=(10000,128))
    effects={}
    for ref in ['baseline','daily']:
        for metric in ['reward_day','fill']:
            x=np.array([[index[f'episode_{s}',k][metric]-index['baseline' if ref=='baseline' else f'daily_{s}',k][metric]
                         for k in KEYS] for s in SEEDS])
            if not np.isfinite(x).all():raise ValueError('finite outcomes required')
            draws=x[seed_draws[:,:,None],key_draws[:,None,:]].mean((1,2))
            effects[f'{ref}:{metric}']=dict(mean=float(x.mean()),seed_means=x.mean(1).tolist(),
                ci9875=np.quantile(draws,[.00625,.99375]).tolist())
    r,f=effects['baseline:reward_day'],effects['baseline:fill']
    primary=r['mean']>=.5 and r['ci9875'][0]>0 and f['ci9875'][0]>=-.005 and min(r['seed_means'])>0
    r,f=effects['daily:reward_day'],effects['daily:fill']
    matched=r['ci9875'][0]>0 and f['ci9875'][0]>=-.005
    return dict(effects=effects,primary_pass=bool(primary),matched_superiority_pass=bool(matched),
        means={l:{m:float(np.mean([index[l,k][m] for k in KEYS])) for m in ['reward_day','fill']} for l in labels})
