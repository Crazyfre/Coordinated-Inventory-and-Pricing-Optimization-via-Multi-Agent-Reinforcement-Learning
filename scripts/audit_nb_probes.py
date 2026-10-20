"""Bounded opt-in diagnostic fits; never creates replacement calibration."""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path
import numpy as np
from scipy.linalg import qr


def design_dependency(X, columns):
    X = np.asarray(X, float)
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    rank = int(np.linalg.matrix_rank(X))
    _, _, pivot = qr(X, pivoting=True, mode='economic')
    null = vh[-1] / np.max(np.abs(vh[-1]))
    return dict(rank=rank, columns=len(columns),drop_columns=[columns[i] for i in pivot[rank:]],
                null_relation={c:float(v) for c,v in zip(columns,null) if abs(v)>1e-7},
                null_residual_max=float(np.max(np.abs(X@null))))


def main():
    import statsmodels.api as sm
    from src.demand.alignment import build_aligned_dataset
    from src.demand.calibration import model_exog
    ap=argparse.ArgumentParser(); ap.add_argument('--output', required=True); args=ap.parse_args()
    out=Path(args.output)
    if out.exists(): raise ValueError('fresh output required')
    frozen=Path('outputs/evaluation_research/calibration/train_only_nb.json'); original=frozen.read_bytes(); params=json.loads(original)
    data=build_aligned_dataset(); records=[]; started=time.perf_counter()
    for sid in data.series_ids:
        if time.perf_counter()-started>120: raise TimeoutError('120s budget')
        tr=data.df[(data.df.series_id==sid)&(data.df.day_index<=1885)&(data.df.price_observed==1)]
        va=data.df[(data.df.series_id==sid)&data.df.day_index.between(1886,1913)]
        X=model_exog(tr); V=model_exog(va); dep=design_dependency(X,list(X.columns))
        # Keep existing parameter/prediction provenance; this is an opt-in rank-identifiable probe only.
        keep=[c for c in X.columns if c not in dep['drop_columns']]
        model=sm.NegativeBinomial(tr.y.to_numpy(float),X[keep],loglike_method='nb2')
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always'); result=model.fit(disp=0,maxiter=300)
        p=np.asarray(result.params); mu=np.exp(V[keep].to_numpy()@p[:-1])
        fp=np.array([params['skus'][sid]['params'][c] for c in X.columns]); fmu=np.exp(V.to_numpy()@fp)
        eig=np.linalg.eigvalsh(-model.hessian(p))
        records.append(dict(series=sid,dependency=dep,converged=bool(result.mle_retvals['converged']),
            warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in ws],
            information_eigen_min=float(eig.min()),validation_mu_max_relative_difference=float(np.max(np.abs(mu-fmu)/fmu)),
            validation_mean_ratio=float(va.y.sum()/mu.sum()),alpha=float(p[-1]),
            frozen_alpha=params['skus'][sid]['alpha']))
    # Instrument only likelihood trial calls for the warning-heavy SKU, exact original default fit.
    sid=data.series_ids[0]; tr=data.df[(data.df.series_id==sid)&(data.df.day_index<=1885)&(data.df.price_observed==1)]
    model=sm.NegativeBinomial(tr.y.to_numpy(float),model_exog(tr),loglike_method='nb2')
    original_ll=model.loglike; trials=[]
    def traced(p):
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always'); value=original_ll(p)
        if ws or not np.isfinite(value): trials.append(dict(finite_loglike=bool(np.isfinite(value)),warnings=[str(w.message) for w in ws]))
        return value
    model.loglike=traced
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always'); r=model.fit(disp=0,maxiter=300)
    with warnings.catch_warnings(record=True) as final:
        warnings.simplefilter('always'); final_ll=original_ll(np.asarray(r.params))
    payload=dict(calibration_sha256=hashlib.sha256(original).hexdigest(),seconds=time.perf_counter()-started,
        test_used=False,default_overwritten=False,rank_probes=records,
        overflow_probe=dict(series=sid,converged=bool(r.mle_retvals['converged']),bad_likelihood_trials=trials,
            final_loglike=float(final_ll),final_warnings=[str(w.message) for w in final],
            other_fit_warnings=[str(w.message) for w in ws]))
    assert frozen.read_bytes()==original
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2)); print(json.dumps(payload,indent=2))


if __name__=='__main__': main()
