"""Validation-only bounded service and train-only optimizer audit; no model writes."""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path
import numpy as np


def conditional_bound(demand, initial, capacity, min_lead):
    """Clairvoyant capacity envelope conditional on a fixed realized price path."""
    demand = np.asarray(demand, float)
    initial = np.asarray(initial, float)
    if demand.ndim != 2 or initial.shape != demand.shape[1:] or np.any(demand < 0) or np.any(initial < 0) or not np.isfinite(demand).all() or not np.isfinite(initial).all() or capacity <= 0 or initial.sum() > capacity or min_lead < 1:
        raise ValueError('invalid bound inputs')
    unavailable = min(min_lead, len(demand))
    sales = np.minimum(initial, demand[:unavailable].sum(axis=0)).sum()
    sales += np.minimum(demand[unavailable:].sum(axis=1), capacity).sum()
    return 1.0 if demand.sum() == 0 else float(sales / demand.sum())


def fit_status(params, retvals, predictions):
    finite = bool(np.isfinite(params).all() and np.isfinite(predictions).all())
    converged = bool(retvals.get('converged', False))
    return dict(finite=finite, converged=converged,
                accepted=bool(finite and converged and params[-1] > 0 and np.all(np.asarray(predictions) > 0)))


def main():
    import torch
    from scipy.stats import t
    from src.demand.alignment import build_aligned_dataset
    from src.demand.calibration import model_exog, nbinom_nll
    from src.experiments.research import make_research_env, training_statistics_view
    from src.experiments.phase4 import heuristic
    from src.baselines.operating import ForecastBaseStock
    from scripts.diagnose_operations import replay, summarize_trace
    import statsmodels.api as sm
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--seed-start', type=int, default=20000)
    ap.add_argument('--calibration', action='store_true')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    if not 1 <= args.episodes <= 20:
        raise ValueError('bounded to 1..20 episodes')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('fresh output required')
    start = time.perf_counter()
    frozen = Path('outputs/evaluation_research/calibration/train_only_nb.json')
    original = frozen.read_bytes()
    params = json.loads(original)
    assert params['fit_end_day'] <= 1885 and params['outcome_statistics_end_day'] <= 1885
    data = build_aligned_dataset()
    aligned = training_statistics_view(data)
    labels = ['ss_markup', 'constant_markup', 'cover6_markup', 'constant_baseprice']
    episodes = []
    demands = {}
    audited_rows = 0
    for label in labels:
        env = make_research_env('S2', split='validation', params=params, aligned=aligned)
        order = ForecastBaseStock(6) if label.startswith('cover6') else lambda e,o: heuristic(e, kind='ss' if label.startswith('ss') else 'constant')
        price = 2 if label.endswith('baseprice') else 4
        with (out/(label+'.jsonl')).open('w') as f:
            for seed in range(args.seed_start, args.seed_start+args.episodes):
                if time.perf_counter()-start > 240:
                    raise TimeoutError('240s service budget')
                rows = replay(env, order, lambda o: price, seed, label)
                demand = np.array([r['demand'] for r in rows])
                if price == 4:
                    if seed in demands:
                        assert np.array_equal(demands[seed], demand), 'same-price pairing failure'
                    demands[seed] = demand
                for i, r in enumerate(rows):
                    assert np.allclose(np.array(r['inventory_start'])-r['sales'], r['inventory_end'])
                    assert np.allclose(np.array(r['sales'])+r['lost_sales'], r['demand'])
                    assert np.allclose(np.array(r['accepted'])+r['clipped'], r['arrivals'])
                    assert sum(r['inventory_start']) <= 500+1e-6
                    if i:
                        assert np.allclose(np.array(rows[i-1]['inventory_end'])+r['accepted'],r['inventory_start'])
                    assert abs(2*r['reward_terms']['revenue']-sum(r['reward_terms'].values())-r['reward']) < 1e-6
                    audited_rows += 1
                    f.write(json.dumps(r)+'\n')
                summary = summarize_trace(rows)
                summary['bound'] = conditional_bound(demand, rows[0]['inventory_start'],500,2)
                for name, sl in [('cold',slice(0,7)),('core',slice(7,21)),('end',slice(21,28))]:
                    d = demand[sl]
                    summary[name+'_capacity_bound'] = float(np.minimum(d.sum(axis=1),500).sum()/max(d.sum(),1))
                episodes.append(summary)
        (out/'episodes.json').write_text(json.dumps(episodes,indent=2))
        print(label, flush=True)
    aggregate = {label:{key:float(np.mean([e[key] for e in episodes if e['label']==label])) for key in episodes[0] if key not in ('label','seed')} for label in labels}
    paired = {}
    ref = [e for e in episodes if e['label']=='ss_markup']
    for label in labels[1:]:
        other = [e for e in episodes if e['label']==label]
        paired[label] = {}
        for key in ('reward_day','fill'):
            delta = np.array([b[key]-a[key] for a,b in zip(ref,other)])
            half = float(t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else None
            paired[label][key] = dict(mean=float(delta.mean()),t95=None if half is None else [float(delta.mean()-half),float(delta.mean()+half)])
    bounds = [e['bound'] for e in ref]
    result = dict(episodes=len(episodes),trace_days=audited_rows,seeds=list(range(args.seed_start,args.seed_start+args.episodes)),scenario='S2',validation=[1886,1913],test_used=False,physics_changed=False,calibration_sha256=hashlib.sha256(original).hexdigest(),aggregates=aggregate,paired=paired,bounds=dict(min=min(bounds),max=max(bounds),mean=float(np.mean(bounds)),below95=sum(b<.95 for b in bounds)),seconds=time.perf_counter()-start)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    if args.calibration:
        audit = []
        for sid in data.series_ids:
            if time.perf_counter()-start > 480:
                raise TimeoutError('480s total budget')
            tr = data.df[(data.df.series_id==sid)&(data.df.day_index<=1885)&(data.df.price_observed==1)]
            va = data.df[(data.df.series_id==sid)&data.df.day_index.between(1886,1913)]
            X = model_exog(tr); y = tr.y.to_numpy(float); V = model_exog(va)
            # Replicate exact default fit path; capture warnings, do not suppress evidence.
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                model = sm.NegativeBinomial(y,X,loglike_method='nb2')
                res = model.fit(disp=0,maxiter=300)
            p = np.asarray(res.params)
            mu = np.exp(V.to_numpy()@p[:-1])
            frozen_p = np.array([params['skus'][sid]['params'][c] for c in X.columns]+[params['skus'][sid]['alpha']])
            with warnings.catch_warnings(record=True) as final_ws:
                warnings.simplefilter('always')
                ll = model.loglike(p)
                score = model.score(p)
                hessian = model.hessian(p)
            eig = np.linalg.eigvalsh(-hessian)
            record = dict(series=sid,n_train=len(tr),train_max=int(tr.day_index.max()),recorded_convergence_status=params['skus'][sid].get('converged','NOT_RECORDED'),status=fit_status(p,res.mle_retvals,mu),warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in ws],final_evaluation_warnings=[str(w.message) for w in final_ws],retvals={k:np.asarray(v).tolist() for k,v in res.mle_retvals.items()},max_parameter_difference=float(np.max(np.abs(p-frozen_p))),final_loglike=float(ll),score_max=float(np.max(np.abs(score))),information_eigen_min=float(eig.min()),design_rank=int(np.linalg.matrix_rank(X)),design_columns=X.shape[1],validation=dict(mu_min=float(mu.min()),mu_max=float(mu.max()),mean_ratio=float(va.y.sum()/mu.sum()),mae=float(np.mean(np.abs(va.y.to_numpy()-mu))),nll=nbinom_nll(va.y.to_numpy(),mu,p[-1])))
            # Same fitted model at frozen final parameters independently identifies final-state failures.
            record['frozen_status'] = fit_status(frozen_p,{},np.exp(V.to_numpy()@frozen_p[:-1]))
            audit.append(record)
            (out/'calibration_audit.json').write_text(json.dumps(audit,indent=2))
            print(sid, record['status'],len(ws),'warnings',flush=True)
    assert frozen.read_bytes()==original, 'frozen artifact changed'
    result['total_seconds']=time.perf_counter()-start
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
