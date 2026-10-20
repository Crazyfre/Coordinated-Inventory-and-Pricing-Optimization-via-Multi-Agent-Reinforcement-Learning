"""Structural, train-only and simulator compatibility gates for opt-in NB v2."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.demand.alignment import build_aligned_dataset
from src.demand.calibration import model_exog
from src.demand.identifiable_calibration import (
    identifiable_exog, fit_identifiable_training_calibration,
)
from scripts.correct_training_calibration import audit_equivalence


@pytest.fixture(scope="module")
def aligned():
    return build_aligned_dataset()


@pytest.fixture(scope="module")
def corrected(aligned):
    return fit_identifiable_training_calibration(aligned)


def test_drop_named_redundancy_and_keep_type_effects(aligned):
    for sid in aligned.series_ids:
        tr = aligned.df[(aligned.df.series_id == sid)
                        & (aligned.df.day_index <= 1885)
                        & (aligned.df.price_observed == 1)]
        old, new = model_exog(tr), identifiable_exog(tr)
        assert np.linalg.matrix_rank(old) == 27 < old.shape[1]
        assert new.shape[1] == np.linalg.matrix_rank(new) == 27
        assert "is_event" not in new
        np.testing.assert_array_equal(old["is_event"], old[
            ["event_cultural", "event_sporting", "event_religious", "event_national"]].sum(axis=1))


def test_converged_identifiable_exports_are_explicitly_compatible(corrected):
    assert corrected["status"] == "research_train_only_identifiable_v2"
    assert corrected["fit_end_day"] == corrected["outcome_statistics_end_day"] == 1885
    for sku in corrected["skus"].values():
        d = sku["fit_diagnostics"]
        assert d["converged"] and d["rank"] == d["columns"] == 27
        assert d["information_eigen_min"] > 0
        assert d["final_evaluation_warning_count"] == 0
        assert "is_event" not in d["fitted_exog"]
        assert sku["params"]["is_event"] == 0.0
        assert np.isfinite(sku["alpha"]) and sku["alpha"] > 0
        assert not any(w["category"] == "HessianInversionWarning" for w in d["fit_warnings"])


def test_future_outcomes_and_unobserved_prices_do_not_enter_fit(aligned, corrected):
    df = aligned.df.copy()
    excluded = (df.day_index > 1885) | (df.price_observed != 1)
    # Poison excluded outcomes, price and all history columns, not just test labels.
    df.loc[excluded, ["y", "hist_7", "hist_28", "log_price_rel"]] = np.nan
    assert fit_identifiable_training_calibration(replace(aligned, df=df)) == corrected


def test_additional_rank_defect_fails_closed(aligned):
    df = aligned.df.copy()
    df["hist_28"] = df["hist_7"]
    with pytest.raises(ValueError, match="corrected design rank"):
        fit_identifiable_training_calibration(replace(aligned, df=df))


def test_validation_and_controlled_simulator_equivalence_do_not_score_test(aligned, corrected):
    path = Path("outputs/evaluation_research/calibration/train_only_nb.json")
    before = path.read_bytes()
    frozen = json.loads(before)
    audit = audit_equivalence(aligned, frozen, corrected)
    assert audit["equivalence_pass"]
    assert len(audit["simulator_comparisons"]) == 15
    assert audit["test_used"] is False
    df = aligned.df.copy()
    df.loc[df.day_index >= 1914, ["y", "hist_7", "hist_28", "log_price_rel"]] = np.nan
    assert audit_equivalence(replace(aligned, df=df), frozen, corrected) == audit
    assert path.read_bytes() == before


def test_version_aware_loader_accepts_candidate_and_rejects_leakage(aligned, corrected, tmp_path):
    from copy import deepcopy
    import hashlib
    from src.demand.identifiable_calibration import (
        load_identifiable_training_calibration, validate_research_training_calibration,
    )
    frozen_bytes = Path("outputs/evaluation_research/calibration/train_only_nb.json").read_bytes()
    frozen = json.loads(frozen_bytes)
    candidate = deepcopy(corrected)
    candidate["correction_audit"] = audit_equivalence(aligned, frozen, candidate)
    candidate["source_frozen_sha256"] = hashlib.sha256(frozen_bytes).hexdigest()
    candidate["correction_audit"]["frozen_sha256"] = candidate["source_frozen_sha256"]
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate))
    loaded = load_identifiable_training_calibration(path, expected_series_order=aligned.series_ids)
    assert validate_research_training_calibration(loaded) == candidate
    assert validate_research_training_calibration(frozen) == frozen
    for key, bad in [("status", "calibrated_v1"),
                     ("outcome_statistics_end_day", 1941), ("fit_end_day", 1913)]:
        invalid = deepcopy(candidate)
        invalid[key] = bad
        with pytest.raises(ValueError):
            validate_research_training_calibration(invalid)
    invalid = deepcopy(candidate)
    invalid["correction_audit"]["maxima"]["simulator_mu_max_relative"] = .5
    with pytest.raises(ValueError, match="equivalence"):
        validate_research_training_calibration(invalid)
    # Pass the distinct version directly to the real factory, without fitting/training.
    from src.experiments.research import make_research_env, training_statistics_view
    env = make_research_env(params=loaded, days=1, aligned=training_statistics_view(aligned))
    obs, _ = env.reset(seed=123)
    assert np.isfinite(obs["replenishment"]).all()


def test_frozen_source_provenance_and_series_order_are_required(aligned):
    from copy import deepcopy
    from scripts.correct_training_calibration import validate_frozen_source
    frozen = json.loads(Path("outputs/evaluation_research/calibration/train_only_nb.json").read_bytes())
    validate_frozen_source(frozen, aligned.series_ids)
    for key, value in [("status", "calibrated_v1"),
                       ("outcome_statistics_end_day", 1941),
                       ("series_order", list(reversed(aligned.series_ids)))]:
        invalid = deepcopy(frozen)
        invalid[key] = value
        with pytest.raises(ValueError, match="provenance"):
            validate_frozen_source(invalid, aligned.series_ids)
