"""Opt-in train-only NB2 v2: remove redundant is_event, retain legacy defaults.

The fitted design has 27 columns. The exported compatibility parameter is_event
is exactly zero, not estimated; existing simulator calendar layout stays intact.
No fallback, warning suppression, test scoring or default-artifact mutation.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.demand.calibration import model_exog, predict_mu, pearson_residuals

FIT_END = 1885
VERSION = "research_train_only_identifiable_v2"


def identifiable_exog(df):
    """Keep named event-type effects; omit their redundant aggregate."""
    return model_exog(df).drop(columns=["is_event"])


def fit_identifiable_training_calibration(data):
    """Return a compatible payload, failing closed on structural/fit problems."""
    train = data.df[(data.df.day_index <= FIT_END) & (data.df.price_observed == 1)]
    skus, residuals = {}, {}
    for sid in data.series_ids:
        tr = train[train.series_id == sid]
        X = identifiable_exog(tr)
        y = tr.y.to_numpy(float)
        rank = int(np.linalg.matrix_rank(X))
        if rank != X.shape[1]:
            raise ValueError(f"{sid}: corrected design rank {rank}/{X.shape[1]}")
        model = sm.NegativeBinomial(y, X, loglike_method="nb2")
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always")
            result = model.fit(disp=0, maxiter=300)
        p = np.asarray(result.params, float)
        with warnings.catch_warnings(record=True) as final_warnings:
            warnings.simplefilter("always")
            ll = model.loglike(p)
            score = model.score(p)
            information = -model.hessian(p)
            eig = np.linalg.eigvalsh(information)
            mu = predict_mu(p[:-1], X)
        converged = bool(result.mle_retvals.get("converged", False))
        if not (converged and np.isfinite(p).all() and p[-1] > 0
                and np.isfinite(ll) and np.isfinite(score).all()
                and np.isfinite(eig).all() and eig.min() > 0
                and np.isfinite(mu).all() and (mu > 0).all()
                and not final_warnings):
            raise ValueError(f"{sid}: convergence/final-evaluation gate failed")
        if any(w.category.__name__ == "HessianInversionWarning" for w in fit_warnings):
            raise ValueError(f"{sid}: information inversion failed")
        params = dict(zip(X.columns, map(float, p[:-1])))
        params["is_event"] = 0.0  # compatibility only; never fitted
        alpha = float(p[-1])
        skus[sid] = dict(
            params=params, alpha=alpha, phi=1 / alpha, n_train_days=len(tr),
            fit_diagnostics=dict(
                converged=converged, fitted_exog=list(X.columns),
                rank=rank, columns=X.shape[1], loglike=float(ll),
                score_max_abs=float(np.abs(score).max()),
                information_eigen_min=float(eig.min()),
                final_evaluation_warning_count=len(final_warnings),
                fit_warnings=[dict(category=w.category.__name__, message=str(w.message))
                              for w in fit_warnings],
            ),
        )
        residuals[sid] = pd.Series(pearson_residuals(y, mu, alpha), index=tr.day_index)
    res = pd.DataFrame(residuals).reindex(columns=data.series_ids).dropna()
    corr = res.corr().to_numpy()
    if not np.isfinite(corr).all():
        raise ValueError("nonfinite residual correlation")
    return dict(
        status=VERSION, fit_end_day=FIT_END, outcome_statistics_end_day=FIT_END,
        feature_correction=dict(dropped_fitted_feature="is_event",
                                compatibility_fixed_parameters={"is_event": 0.0},
                                reason="is_event equals sum of four event-type indicators"),
        fit_mask="day_index <= 1885 AND price_observed == 1",
        series_order=list(data.series_ids), skus=skus,
        residual_correlation=dict(matrix=corr.tolist(), n_common_days=len(res)),
        test_used=False, default_overwritten=False,
    )


def load_identifiable_training_calibration(path, *, expected_series_order=None):
    """Explicit v2 loader; no legacy fallback or automatic default adoption."""
    import json
    from pathlib import Path
    return validate_identifiable_training_calibration(
        json.loads(Path(path).read_text()), expected_series_order=expected_series_order)


def validate_identifiable_training_calibration(payload, *, expected_series_order=None):
    """Validate a compatible v2 payload without changing its version."""
    if (payload.get("status") != VERSION
            or payload.get("fit_end_day") != FIT_END
            or payload.get("outcome_statistics_end_day") != FIT_END
            or payload.get("feature_correction", {}).get("dropped_fitted_feature") != "is_event"
            or not payload.get("correction_audit", {}).get("equivalence_pass")):
        raise ValueError("invalid identifiable-v2 provenance/equivalence gate")
    audit = payload["correction_audit"]
    source_hash = payload.get("source_frozen_sha256", "")
    if (len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash)
            or audit.get("frozen_sha256") != source_hash
            or audit.get("test_used") is not False
            or payload.get("test_used") is not False
            or payload.get("default_overwritten") is not False):
        raise ValueError("invalid v2 source/evaluation metadata")
    required = {"validation_mu_max_relative": .001, "simulator_mu_max_relative": .001,
                "alpha_max_relative": .01, "correlation_max_absolute": .001}
    for key, limit in required.items():
        value = audit.get("maxima", {}).get(key, float("nan"))
        if (not np.isfinite(value) or value < 0 or value > limit
                or audit.get("tolerances", {}).get(key) != limit):
            raise ValueError("invalid v2 predictive-equivalence measurements")
    order = payload.get("series_order", [])
    if (not order or len(set(order)) != len(order)
            or set(payload.get("skus", {})) != set(order)
            or (expected_series_order is not None and order != list(expected_series_order))):
        raise ValueError("calibration series order mismatch")
    for sid in order:
        sku = payload["skus"][sid]
        d = sku.get("fit_diagnostics", {})
        fitted = d.get("fitted_exog", [])
        if (not d.get("converged") or not fitted or "is_event" in fitted
                or d.get("rank") != len(fitted) or d.get("columns") != len(fitted)
                or sku["params"].get("is_event") != 0.0
                or not np.isfinite(d.get("information_eigen_min", 0))
                or d.get("information_eigen_min", 0) <= 0
                or d.get("final_evaluation_warning_count") != 0
                or not np.isfinite(list(sku["params"].values())).all()
                or not np.isfinite(sku["alpha"]) or sku["alpha"] <= 0
                or not np.isclose(sku["alpha"] * sku["phi"], 1.0)):
            raise ValueError(f"{sid}: invalid corrected fit diagnostics")
    corr = np.asarray(payload["residual_correlation"]["matrix"], float)
    if (corr.shape != (len(order), len(order)) or not np.isfinite(corr).all()
            or not np.allclose(corr, corr.T) or not np.allclose(np.diag(corr), 1)
            or np.linalg.eigvalsh(corr).min() < -1e-8):
        raise ValueError("invalid residual correlation")
    return payload


def validate_research_training_calibration(payload, *, expected_series_order=None):
    """Accept frozen legacy or explicitly identified v2; reject leakage/unknown versions."""
    if (payload.get("fit_end_day") != FIT_END
            or payload.get("outcome_statistics_end_day") != FIT_END):
        raise ValueError("train-only calibration provenance required")
    if payload.get("status") == VERSION:
        return validate_identifiable_training_calibration(
            payload, expected_series_order=expected_series_order)
    if payload.get("status") != "research_train_only":
        raise ValueError("unsupported research calibration version")
    if expected_series_order is not None and payload.get("series_order") != list(expected_series_order):
        raise ValueError("calibration series order mismatch")
    return payload
