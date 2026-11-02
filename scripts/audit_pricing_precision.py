"""Descriptive variance audit of existing paired pricing futures; zero simulation."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.train_compact_residual import digest, write_json


def run(source, output):
    root, out = Path(source), Path(output)
    summary = json.loads((root / 'summary.json').read_text())
    proof = json.loads((root / 'verification.json').read_text())
    assert proof['status'] == 'passed'
    for p, h in summary['trace_hashes'].items():
        assert digest(root / p) == h
    states, gains = [], []
    with (root / 'states.jsonl').open() as handle:
        for line in handle:
            r = json.loads(line)
            v = np.array([[b['value'] for b in d['branches']] for d in r['draws']])
            assert v.shape == (4, 5) and np.isfinite(v).all()
            gains.append(v[:, :4] - v[:, 4, None])
            states.append([r['episode'], r['day'], r['sku']])
    g = np.array(gains)
    assert g.shape == (64, 4, 4) and len({tuple(s) for s in states}) == 64
    results = {}
    for action in range(4):
        x = g[:, :, action]
        means = x.mean(1)
        variances = x.var(1, ddof=1)
        # Independently reconstruct each sample variance through pair differences.
        check = np.array([sum((row[i]-row[j])**2 for i in range(4)
                              for j in range(i+1, 4))/12 for row in x])
        assert np.allclose(variances, check, rtol=1e-12, atol=1e-10)
        first, second = x[:, :2].mean(1), x[:, 2:].mean(1)
        results[str(action)] = dict(
            mean_discounted_suffix_gain=float(means.mean()),
            median_state_gain=float(np.median(means)),
            positive_four_draw_means=int((means > 0).sum()),
            positive_in_both_halves=int(((first > 0) & (second > 0)).sum()),
            positive_first_only=int(((first > 0) & (second <= 0)).sum()),
            positive_second_only=int(((first <= 0) & (second > 0)).sum()),
            within_state_sd_quantiles=np.quantile(np.sqrt(variances), [.25, .5, .75, .9]).tolist(),
            two_draw_se_quantiles=np.quantile(np.sqrt(variances / 2), [.25, .5, .75, .9]).tolist(),
            estimated_draws_for_se_one_quantiles=np.quantile(np.maximum(2, np.ceil(variances)), [.5, .9]).tolist(),
            estimated_draws_for_se_two_quantiles=np.quantile(np.maximum(2, np.ceil(variances / 4)), [.5, .9]).tolist())
    result = dict(status='complete', states=64, futures_per_state=4,
        units='discounted remaining-episode reward difference, not reward/day',
        actions=results, new_simulator_days=0, new_policy_fits=0,
        variance_identity_verified=True,
        limitation='Post-result descriptive audit. Four-draw variance estimates are noisy; sample-size projections are not power guarantees. States within each original window are dependent. Positive sample means are not verified conditional opportunities.',
        provenance={p: digest(root / p) for p in ['summary.json', 'verification.json', 'states.jsonl']},
        script_sha256=digest(__file__))
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'audit.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run(a.source, a.output)
