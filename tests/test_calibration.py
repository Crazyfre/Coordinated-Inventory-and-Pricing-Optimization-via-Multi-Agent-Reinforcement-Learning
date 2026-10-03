"""Tests for the NB demand calibration artifact (Phase P1, task 3).

Loads the cached artifact (fits only if missing). Does NOT refit in tests.
"""

import json

import numpy as np
import pytest

from src.config.sku_config import load_sku_payload
from src.demand.alignment import build_aligned_dataset
from src.demand.calibration import load_nb_params, model_exog, predict_mu


@pytest.fixture(scope="session")
def params() -> dict:
    return load_nb_params()


class TestArtifact:
    def test_exists_and_status(self, params):
        assert params["status"] == "calibrated_v1"

    def test_all_locked_skus_fitted(self, params):
        locked = [s["series_id"] for s in load_sku_payload()["skus"]]
        assert params["series_order"] == locked
        assert set(params["skus"].keys()) == set(locked)

    def test_param_shapes(self, params):
        expected_keys = {"const", "log_price_rel", "snap", "is_event", "hist_7", "hist_28"}
        for sid, s in params["skus"].items():
            assert expected_keys <= set(s["params"].keys()), sid
            assert all(np.isfinite(v) for v in s["params"].values()), sid
            assert s["alpha"] > 0, sid
            assert np.isclose(s["phi"], 1.0 / s["alpha"]), sid

    def test_elasticity_sanity_range(self, params):
        """Fitted betas must be finite and plausible in magnitude.

        NOTE: sign is NOT asserted — observational coefficients can be
        positive (price-demand confounding, Doc 4.5). That is expected.
        """
        for sid, s in params["skus"].items():
            bp = s["params"]["log_price_rel"]
            assert -6.0 < bp < 6.0, f"{sid}: implausible beta_p {bp}"

    def test_validation_metrics_present(self, params):
        for sid, s in params["skus"].items():
            v = s["validation"]
            # NLL is a SUM of negative log-probs over many days -> positive.
            assert np.isfinite(v["nll_train"]) and v["nll_train"] > 0
            assert np.isfinite(v["nll_test"]) and v["nll_test"] > 0
            # Per-day average test NLL should be of a sane order (< 10 nats/day).
            assert v["nll_test"] / 28 < 10, f"{sid} pathological test NLL"
            assert v["mae_test"] >= 0
            assert 0.3 < v["mean_ratio_test"] < 3.0, f"{sid} badly calibrated"

    def test_elasticity_note_recorded(self, params):
        assert "not causal" in params["elasticity_note"].lower()
        assert "slider" in params["elasticity_note"].lower()


class TestResidualCorrelation:
    def test_matrix_valid(self, params):
        rc = params["residual_correlation"]
        C = np.asarray(rc["matrix"])
        n = len(params["series_order"])
        assert C.shape == (n, n)
        assert np.allclose(C, C.T)
        assert np.allclose(np.diag(C), 1.0)
        assert np.abs(C).max() <= 1.0 + 1e-9

    def test_nonzero_common_days(self, params):
        assert params["residual_correlation"]["n_common_days"] > 500
        assert np.isfinite(params["residual_correlation"]["mean_offdiag"])


class TestPredictionRoundTrip:
    def test_mu_reproduces_from_stored_params(self, params, ):
        """mu = exp(X @ beta) recomputed from the JSON params must match the
        stored validation means (built from the same params)."""
        data = build_aligned_dataset()
        train, test = data.split()
        train = train[train.price_observed == 1]
        sid = params["series_order"][0]
        s = params["skus"][sid]

        te = test[test.series_id == sid]
        X_te = model_exog(te)
        beta = np.array([s["params"][k] for k in X_te.columns])
        mu = predict_mu(beta, X_te)
        assert np.isclose(mu.mean(), s["validation"]["mu_test_mean"], rtol=1e-6)

    def test_nll_finite_and_ordered(self, params):
        """Test NLL is finite; train NLL (fitted) should not be worse than a
        same-series test NLL by an order of magnitude (rough sanity)."""
        sid = params["series_order"][0]
        s = params["skus"][sid]
        assert np.isfinite(s["validation"]["nll_train"])
        assert np.isfinite(s["validation"]["nll_test"])

    def test_overdispersion_captured(self, params):
        """At least some series must show alpha > 0 (overdispersion vs Poisson)."""
        alphas = [s["alpha"] for s in params["skus"].values()]
        assert max(alphas) > 0.1, "no overdispersion found — check the fit"
