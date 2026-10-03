"""P1 task 3: Negative Binomial demand calibration (Doc 4.4) for locked SKUs.

Per-SKU model (NB2, log link):
    D_(i,t) ~ NegBin(mean = mu_(i,t), dispersion phi_i)
    log(mu_(i,t)) = b0 + b_p * log(p/p_base) + b_x^T x_t + b_h^T H_t

    - Var(mu) = mu + alpha * mu^2,  alpha = 1/phi  (NB2 parameterization)
    - fitted jointly by MLE (statsmodels NegativeBinomial)

Causal discipline (Doc 4.5): the fitted b_p is a PREDICTIVE coefficient from
observational data — it is NOT treated as the causal elasticity. The
simulator's price response uses a CONTROLLED elasticity slider; b_p values
are stored only as a sanity range for that slider.

Fit mask: train days only (Doc 4.3 split) AND price_observed == 1
(M5 has no price records for some early weeks — fitting elasticity on
filled prices would contaminate the estimate).

Residual structure: standardized Pearson residuals on common train days,
correlated across the 10 SKUs -> drives the S5 correlated-demand scenario.

Backtest: one-step-ahead on the held-out 28 test days, with true history
features (hist_7/hist_28 as computed in the aligned table).

Artifacts: data/processed/nb_params.json (cached; delete to refit).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

import statsmodels.api as sm

from src.demand.alignment import AlignedData, EVENT_TYPES, build_aligned_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "data" / "processed" / "nb_params.json"

# Reference levels dropped to avoid the dummy-variable trap: dow_1, month_1.
EXOG_BASE = (
    ["log_price_rel"]
    + [f"dow_{w}" for w in range(2, 8)]
    + [f"month_{m}" for m in range(2, 13)]
    + ["snap", "is_event", "year_idx"]
    + [f"event_{t.lower()}" for t in EVENT_TYPES]
)
EXOG_HIST = ["hist_7", "hist_28"]


def model_exog(df: pd.DataFrame) -> pd.DataFrame:
    """Design matrix (intercept + price + calendar + history)."""
    X = df[EXOG_BASE + EXOG_HIST].astype(float).copy()
    X.insert(0, "const", 1.0)
    return X


def nbinom_nll(y: np.ndarray, mu: np.ndarray, alpha: float) -> float:
    """NB2 negative log-likelihood: Var = mu + alpha*mu^2."""
    n = 1.0 / alpha
    p = n / (n + mu)
    return float(-sps.nbinom.logpmf(y, n, p).sum())


def _fit_series(y: np.ndarray, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """MLE fit; falls back to Poisson start params if convergence struggles."""
    try:
        res = sm.NegativeBinomial(y, X, loglike_method="nb2").fit(disp=0, maxiter=300)
    except Exception:  # noqa: BLE001 - fallback path
        pois = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        res = sm.NegativeBinomial(y, X, loglike_method="nb2").fit(
            start_params=np.append(np.asarray(pois.params, dtype=float), 1.0),
            disp=0, maxiter=500,
        )
    all_params = np.asarray(res.params, dtype=float)
    beta = all_params[:-1]
    alpha = float(all_params[-1])
    assert alpha > 0, "fitted alpha must be positive"
    return beta, alpha


def predict_mu(beta: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    return np.exp(X.to_numpy() @ beta)


def pearson_residuals(y, mu, alpha) -> np.ndarray:
    var = mu + alpha * mu**2
    return (y - mu) / np.sqrt(var)


# ---------------------------------------------------------------------------
# fit-all + artifact
# ---------------------------------------------------------------------------

def fit_all(data: AlignedData | None = None, force: bool = False) -> dict:
    """Fit all 10 SKUs; cache the artifact. Returns the params dict."""
    if ARTIFACT.exists() and not force:
        return json.loads(ARTIFACT.read_text())

    data = data or build_aligned_dataset()
    train, test = data.split()
    train = train[train.price_observed == 1]  # elasticity needs real prices
    series_ids = list(data.series_ids)

    # Residuals need aligned days across series: pivot to (day, series).
    res_cols: dict[str, np.ndarray] = {}

    skus: dict[str, dict] = {}
    for sid in series_ids:
        tr = train[train.series_id == sid]
        te = test[test.series_id == sid]
        X_tr, X_te = model_exog(tr), model_exog(te)
        y_tr, y_te = tr["y"].to_numpy(float), te["y"].to_numpy(float)

        beta, alpha = _fit_series(y_tr, X_tr)

        mu_tr, mu_te = predict_mu(beta, X_tr), predict_mu(beta, X_te)
        res_cols[sid] = pd.Series(pearson_residuals(y_tr, mu_tr, alpha), index=tr["day_index"])

        skus[sid] = {
            "params": {k: float(v) for k, v in zip(X_tr.columns, beta)},
            "alpha": alpha,
            "phi": 1.0 / alpha,
            "n_train_days": int(len(tr)),
            "validation": {
                "nll_train": nbinom_nll(y_tr, mu_tr, alpha),
                "nll_test": nbinom_nll(y_te, mu_te, alpha),
                "mae_test": float(np.abs(y_te - mu_te).mean()),
                "mean_ratio_test": float(y_te.sum() / mu_te.sum()),  # ~1 = well calibrated
                "mu_test_mean": float(mu_te.mean()),
                "y_test_mean": float(y_te.mean()),
            },
        }

    # Joint residual correlation on common train days (S5 scenario driver).
    res_df = pd.DataFrame(res_cols).dropna()
    corr = res_df.corr().to_numpy()
    off_diag = corr[~np.eye(len(corr), dtype=bool)]

    payload = {
        "status": "calibrated_v1",
        "model": "NB2 log-link per-SKU MLE (statsmodels NegativeBinomial)",
        "fit_mask": "train days (day_index < test_start) AND price_observed == 1",
        "dispersion_note": "alpha = 1/phi; Var = mu + alpha*mu^2",
        "elasticity_note": (
            "fitted log_price_rel coefficients are PREDICTIVE, not causal "
            "(Doc 4.5); used only as a sanity range for the simulator's "
            "controlled elasticity slider"
        ),
        "series_order": series_ids,
        "skus": skus,
        "residual_correlation": {
            "matrix": corr.round(4).tolist(),
            "mean_offdiag": float(off_diag.mean()),
            "max_abs_offdiag": float(np.abs(off_diag).max()),
            "n_common_days": int(len(res_df)),
        },
    }
    ARTIFACT.write_text(json.dumps(payload, indent=2))
    return payload


def load_nb_params() -> dict:
    if not ARTIFACT.exists():
        raise FileNotFoundError(f"{ARTIFACT} missing — run calibration first")
    return json.loads(ARTIFACT.read_text())


# ---------------------------------------------------------------------------
# CLI report
# ---------------------------------------------------------------------------

def _report(payload: dict) -> None:
    print(f"model: {payload['model']}")
    print(f"fit mask: {payload['fit_mask']}\n")
    print(f"{'series':<24}{'beta_p':>8}{'alpha':>8}{'phi':>8}"
          f"{'MAE_te':>8}{'y_mean':>8}{'mu_mean':>8}{'ratio':>7}")
    for sid, s in payload["skus"].items():
        v = s["validation"]
        print(f"{sid:<24}{s['params']['log_price_rel']:>8.3f}{s['alpha']:>8.3f}"
              f"{s['phi']:>8.2f}{v['mae_test']:>8.2f}{v['y_test_mean']:>8.1f}"
              f"{v['mu_test_mean']:>8.1f}{v['mean_ratio_test']:>7.2f}")
    rc = payload["residual_correlation"]
    print(f"\nresidual corr (train): mean off-diag {rc['mean_offdiag']:+.3f}, "
          f"max |corr| {rc['max_abs_offdiag']:.3f} over {rc['n_common_days']} common days")
    betas = [s["params"]["log_price_rel"] for s in payload["skus"].values()]
    print(f"beta_p range: [{min(betas):+.3f}, {max(betas):+.3f}] "
          f"(sanity range for the elasticity slider, NOT causal)")


if __name__ == "__main__":
    _report(fit_all())
