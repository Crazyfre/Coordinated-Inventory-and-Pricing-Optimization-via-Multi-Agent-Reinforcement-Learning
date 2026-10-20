"""Versioned, bounded train-only correction with predeclared validation gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

# Keep the fit bounded without taking worker pools from other experiments.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from src.demand.alignment import build_aligned_dataset
from src.demand.calibration import model_exog, nbinom_nll
from src.demand.identifiable_calibration import fit_identifiable_training_calibration

TOLERANCES = dict(validation_mu_max_relative=0.001,
                  simulator_mu_max_relative=0.001,
                  alpha_max_relative=0.01,
                  correlation_max_absolute=0.001)


def relative_difference(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (b <= 0).any():
        raise ValueError("invalid comparison predictions")
    return float(np.max(np.abs(a - b) / b))


def audit_equivalence(data, frozen, corrected):
    """Validation only: predictive GLM and actual controlled-elasticity simulator."""
    from dataclasses import replace
    from src.experiments.research import training_statistics_view
    from src.demand.simulator import CalibratedDemandSimulator
    validate_frozen_source(frozen, data.series_ids)
    if corrected.get("series_order") != list(data.series_ids):
        raise ValueError("corrected calibration series order mismatch")
    records = []
    for sid in data.series_ids:
        va = data.df[(data.df.series_id == sid) & data.df.day_index.between(1886, 1913)]
        X = model_exog(va)
        def mu(payload):
            return np.exp(X.to_numpy() @ np.array(
                [payload["skus"][sid]["params"][c] for c in X.columns]))
        old, new = mu(frozen), mu(corrected)
        a0, a1 = frozen["skus"][sid]["alpha"], corrected["skus"][sid]["alpha"]
        y = va.y.to_numpy(float)
        records.append(dict(
            series=sid, validation_mu_max_relative_difference=relative_difference(new, old),
            alpha_relative_difference=abs(a1 - a0) / a0,
            old_mean_ratio=float(y.sum() / old.sum()),
            corrected_mean_ratio=float(y.sum() / new.sum()),
            old_nll=nbinom_nll(y, old, a0), corrected_nll=nbinom_nll(y, new, a1),
            old_mae=float(np.abs(y - old).mean()),
            corrected_mae=float(np.abs(y - new).mean()),
        ))
    # Reuse the canonical anchor/history redaction; omit test calendar entirely.
    aligned = training_statistics_view(replace(
        data, df=data.df[data.df.day_index <= 1913].copy()))
    comparisons = []
    for elasticity in (-0.5, -1.5, -3.0):
        oldsim = CalibratedDemandSimulator(nb_params=frozen, aligned=aligned,
                         start_day_index=1886, elasticity=elasticity, shock_std=0)
        newsim = CalibratedDemandSimulator(nb_params=corrected, aligned=aligned,
                         start_day_index=1886, elasticity=elasticity, shock_std=0)
        oldsim.mu_for(1, oldsim._base_price_vector())
        for multiplier in (0.8, 0.9, 1.0, 1.1, 1.2):
            old = np.stack([oldsim.mu_for(day, oldsim._base_price_vector() * multiplier)
                            for day in range(1, 29)])
            new = np.stack([newsim.mu_for(day, newsim._base_price_vector() * multiplier)
                            for day in range(1, 29)])
            comparisons.append(dict(elasticity=elasticity, price_multiplier=multiplier,
                                    max_relative_difference=relative_difference(new, old)))
    corr_delta = float(np.max(np.abs(
        np.asarray(corrected["residual_correlation"]["matrix"])
        - np.asarray(frozen["residual_correlation"]["matrix"]))))
    maxima = dict(
        validation_mu_max_relative=max(r["validation_mu_max_relative_difference"] for r in records),
        simulator_mu_max_relative=max(r["max_relative_difference"] for r in comparisons),
        alpha_max_relative=max(r["alpha_relative_difference"] for r in records),
        correlation_max_absolute=corr_delta,
    )
    return dict(
        validation_days=[1886, 1913], test_used=False, per_sku=records,
        simulator_comparisons=comparisons, tolerances=TOLERANCES, maxima=maxima,
        equivalence_pass=all(maxima[k] <= limit for k, limit in TOLERANCES.items()),
        note="Equivalence to frozen predictions is not predictive accuracy acceptance.",
    )


def validate_frozen_source(frozen, series_order):
    if (frozen.get("status") != "research_train_only"
            or frozen.get("fit_end_day") != 1885
            or frozen.get("outcome_statistics_end_day") != 1885
            or frozen.get("series_order") != list(series_order)
            or set(frozen.get("skus", {})) != set(series_order)):
        raise ValueError("expected matching frozen train-only calibration provenance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen", type=Path,
        default=Path("outputs/evaluation_research/calibration/train_only_nb.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output directory required; existing results preserved")
    original = args.frozen.read_bytes()
    legacy_path = Path("data/processed/nb_params.json")
    legacy = legacy_path.read_bytes() if legacy_path.exists() else None
    frozen = json.loads(original)
    if (frozen.get("status") != "research_train_only"
            or frozen.get("fit_end_day") != 1885
            or frozen.get("outcome_statistics_end_day") != 1885):
        raise ValueError("expected frozen train-only calibration provenance")
    args.output.mkdir(parents=True)
    # Written before fitting/evaluation so thresholds are never chosen after results.
    (args.output / "protocol.json").write_text(json.dumps(dict(
        fit_days=[1, 1885], validation_days=[1886, 1913], test_used=False,
        dropped_feature="is_event", tolerances=TOLERANCES,
        simulator_elasticities=[-0.5, -1.5, -3.0],
        simulator_price_multipliers=[0.8, 0.9, 1.0, 1.1, 1.2],
        frozen_sha256=hashlib.sha256(original).hexdigest(),
    ), indent=2))
    started = time.perf_counter()
    data = build_aligned_dataset()
    validate_frozen_source(frozen, data.series_ids)
    corrected = fit_identifiable_training_calibration(data)
    audit = audit_equivalence(data, frozen, corrected)
    if args.frozen.read_bytes() != original:
        raise RuntimeError("frozen artifact changed")
    if legacy is not None and legacy_path.read_bytes() != legacy:
        raise RuntimeError("legacy default artifact changed")
    audit.update(seconds=time.perf_counter() - started,
                 frozen_sha256=hashlib.sha256(original).hexdigest(),
                 legacy_sha256=None if legacy is None else hashlib.sha256(legacy).hexdigest(),
                 default_overwritten=False)
    (args.output / "equivalence_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False))
    if not audit["equivalence_pass"]:
        raise RuntimeError("predeclared predictive-equivalence gate failed; candidate not published")
    corrected["correction_audit"] = audit
    corrected["source_frozen_sha256"] = hashlib.sha256(original).hexdigest()
    (args.output / "train_only_nb_identifiable_v2.json").write_text(
        json.dumps(corrected, indent=2, allow_nan=False))
    print(json.dumps(dict(status=corrected["status"], seconds=audit["seconds"],
                         equivalence_pass=audit["equivalence_pass"], maxima=audit["maxima"],
                         artifact=str(args.output / "train_only_nb_identifiable_v2.json")), indent=2))


if __name__ == "__main__":
    main()
