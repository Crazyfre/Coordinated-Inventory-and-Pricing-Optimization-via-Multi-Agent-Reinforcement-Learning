"""Archive-only precision planning; no new simulations or model fitting."""
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import norm,t
from scripts.train_rollout_correction import digest


def plan():
    path=Path('outputs/averaged_targets/matched_v1/summary.json')
    prior=json.loads(path.read_text());models=prior['result']['models'];rows={}
    # Planning effects are explicitly chosen, not claimed to be real effects.
    for comparator,effect in [('single',20.),('zero',5.)]:
        delta=np.array(models['averaged']['episode_mse'])-np.array(models[comparator]['episode_mse'])
        sd=float(delta.std(ddof=1));n=8
        while (t.ppf(.975,n-1)+norm.ppf(.8))*sd/math.sqrt(n)>effect:n+=1
        rows[comparator]=dict(pilot_mean=float(delta.mean()),pilot_sd=sd,planning_reduction=effect,
                             approximate_80pct_power_n=n)
    n=8*math.ceil(max(64,max(r['approximate_80pct_power_n'] for r in rows.values()))/8)
    # Increase mean-target precision: pilot action-weighted var(mean16)=15.6725.
    futures=32
    fit_episodes=64
    empirical_days=prior['counts']['fit']['branch_days']/16*fit_episodes+prior['counts']['assessment']['branch_days']/8*n*(futures/16)
    overhead=(fit_episodes+n)*98
    maximum_days=(fit_episodes*8+n*futures)*175+overhead
    result=dict(source=str(path),source_sha256=digest(path),comparisons=rows,assessment_episodes=n,
                fit_episodes=fit_episodes,fit_futures=8,assessment_futures=futures,
                pilot_mean_target_noise=prior['result']['mean_target_noise'],
                projected_mean_target_noise=prior['result']['mean_target_noise']*16/futures,
                empirical_projected_days=empirical_days+overhead,max_prediction_days=maximum_days,
                projected_seconds=(empirical_days+overhead)*prior['seconds']/34048,
                limitations='Pilot SD from8 episodes is unstable; t/normal planning approximation, not guaranteed power. One fitted seed does not estimate training variability.')
    out=Path('outputs/final_rollout_study');out.mkdir(parents=True,exist_ok=True)
    (out/'budget_plan.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':plan()
