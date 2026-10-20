"""Prospectively fixed crossed statistics for the adopted final study."""
import numpy as np
from src.eval.review_metrics import return_summary,success_interval

SEEDS=[211,222,233,244,255,266,277,288,299,310]
KEYS=list(range(2100000,2100050))
CONDITIONS=['main','pooled','independent']
BASELINES=['full7_5','projected','constant','ss']


def effect(values,coverage,seed):
    x=np.asarray(values,float)
    if x.shape!=(10,50) or not np.isfinite(x).all():raise ValueError('finite10-seed by50-scenario matrix required')
    rng=np.random.default_rng(seed);rs=rng.integers(10,size=(10000,10));cs=rng.integers(50,size=(10000,50))
    draws=x[rs[:,:,None],cs[:,None,:]].mean((1,2));alpha=(1-coverage)/2
    return dict(mean=float(x.mean()),interval=np.quantile(draws,[alpha,1-alpha]).tolist(),
                coverage=coverage,seed_means=x.mean(1).tolist())


def analyze(rows):
    labels=[f'{condition}_{s}' for condition in CONDITIONS for s in SEEDS]+BASELINES
    index={(r['label'],r['scenario']):r for r in rows}
    required={(label,key) for label in labels for key in KEYS}
    if len(rows)!=1700 or len(index)!=len(rows) or index.keys()!=required:
        raise ValueError('exact complete34-controller by50-scenario paired grid required')
    for row in rows:
        if not np.isfinite(row['reward_day']) or not 0<=row['fill']<=1:raise ValueError('invalid reward/fill')
    def contrast(reference,metric):
        return np.array([[index[f'main_{s}',k][metric]-index[f'{reference}_{s}' if reference in CONDITIONS else reference,k][metric]
            for k in KEYS] for s in SEEDS])
    primary={metric:effect(contrast('full7_5',metric),.975,20261023) for metric in ['reward_day','fill']}
    secondary={f'{reference}_{metric}':effect(contrast(reference,metric),.9875,20261024)
               for reference in ['pooled','independent'] for metric in ['reward_day','fill']}
    exploratory={f'{reference}_{metric}':effect(contrast(reference,metric),.95,20261025)
                 for reference in BASELINES for metric in ['reward_day','fill']}
    all_positive=bool(np.all(np.array(primary['reward_day']['seed_means'])>0))
    gate=bool(primary['reward_day']['interval'][0]>0 and primary['fill']['interval'][0]>=-.005 and all_positive)
    means={label:{metric:float(np.mean([index[label,k][metric] for k in KEYS])) for metric in ['reward_day','fill']}
           for label in labels}
    distributions={label:dict(episodic_return=return_summary([28*index[label,k]['reward_day'] for k in KEYS]),
        success=success_interval([index[label,k]['fill']>=.95 for k in KEYS])) for label in labels}
    return dict(primary=primary,secondary=secondary,exploratory=exploratory,every_seed_reward_positive=all_positive,
        gate=gate,means=means,distributions=distributions,
        scope='Primary/secondary inferential families apply only to predeclared S2; other conditions exploratory')
