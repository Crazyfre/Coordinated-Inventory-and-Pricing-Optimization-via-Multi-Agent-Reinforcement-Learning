"""Independent archive reconstruction of parameter-space development results."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.train_compact_residual import digest, write_json


def verify(output):
    out = Path(output); load = lambda p: json.loads(p.read_text())
    manifest, summary = load(out/'manifest.json'), load(out/'summary.json')
    for path, value in manifest['hashes'].items():
        assert digest(path) == value
    for name, value in summary['trace_hashes'].items():
        assert digest(out/name) == value
    parameters = {}; expected_scores = {}
    for arm, first_key, dimension in [('sku', 1100000, 10), ('arrival', 1110000, 21)]:
        theta = np.r_[np.full(10, 7.5), np.zeros(dimension-10)]
        sigma = np.r_[np.full(10, .75), np.full(dimension-10, .5)]
        m, v = np.zeros(dimension), np.zeros(dimension); rng = np.random.default_rng(101)
        history = load(out/f'{arm}_seed101_history.json'); assert len(history) == 24
        for u, update in enumerate(history):
            e = rng.normal(size=(8, dimension))
            np.testing.assert_array_equal(e, update['epsilon'])
            np.testing.assert_array_equal(theta, update['mean_before'])
            diffs = []
            for j, record in enumerate(update['records']):
                assert record['key'] == first_key+u*8+j
                for sign, direction in [('plus', 1), ('minus', -1)]:
                    label = f'{arm}_101_u{u}_p{j}_{sign}'
                    parameters[label] = theta+direction*sigma*e[j]
                    expected_scores[label] = record[sign]
                diffs.append(record['plus']['discounted_day']-record['minus']['discounted_day'])
            np.testing.assert_array_equal(diffs, update['differences'])
            gradient = np.mean(np.array(diffs)[:, None]*e/(2*sigma), axis=0)
            np.testing.assert_array_equal(gradient, update['gradient'])
            m = .9*m+.1*gradient; v = .999*v+.001*gradient**2
            theta += .1*(m/(1-.9**(u+1)))/(np.sqrt(v/(1-.999**(u+1)))+1e-8)
            theta[:10] = theta[:10].clip(0, 16); theta[10:] = theta[10:].clip(-8, 8)
            np.testing.assert_array_equal(theta, update['state']['mean'])
        np.testing.assert_array_equal(theta, load(out/f'{arm}_seed101_final.json')['mean'])
        parameters[arm] = theta
    parameters['full7_5'] = np.full(10, 7.5)
    groups = {}
    for filename in ['training_traces.jsonl', 'development_traces.jsonl']:
        with (out/filename).open() as handle:
            for line in handle:
                row = json.loads(line); key = (row['label'], row['scenario'])
                groups.setdefault(key, []).append(row)
                terms = row['info']['reward_terms']
                total = terms['revenue']-sum(terms[k] for k in ['ordering_cost', 'holding_cost', 'stockout_cost', 'price_change_cost'])
                assert abs(total-row['reward']) < 1e-8 and row['info']['total_on_hand'] <= 500+1e-8
                p = parameters.get(row['label'])
                if row['label'].startswith('initial_'):
                    dim = 10 if row['label'] == 'initial_sku' else 21
                    rg = np.random.default_rng(np.random.SeedSequence([101, row['scenario'], 707]))
                    p = np.r_[np.full(10, 7.5), np.zeros(dim-10)]+np.r_[np.full(10, .75), np.full(dim-10, .5)]*rg.normal(size=dim)
                if p is not None:
                    x, p = np.asarray(row['obs'], np.float32), np.asarray(p, np.float32)
                    pipe = x[10:80].reshape(10, 7); cover = p[:10].copy()
                    if len(p) == 21:
                        cover += p[10:20]*pipe[:, 3:].sum(1)/(1+pipe.sum(1))+p[20]*np.float32((29-row['day'])/28-.5)
                    action = np.ceil(np.maximum(cover.clip(0, 20)*x[-10:]-(x[:10]+pipe.sum(1)), 0)/5).clip(0, 40).astype(int)
                    np.testing.assert_array_equal(action, row['actions'])
    recomputed = {}
    for key, rows in groups.items():
        assert [r['day'] for r in rows] == list(range(1, 29))
        scenario = key[1]
        start = int(np.random.default_rng(np.random.SeedSequence([scenario, 314159])).integers(15, 1859))
        assert all(r['start'] == start for r in rows)
        demand = sum(np.asarray(r['info']['demand']).sum() for r in rows)
        recomputed[key] = dict(reward_day=sum(r['reward'] for r in rows)/28,
                              discounted_day=sum(.99**i*r['reward'] for i, r in enumerate(rows))/28,
                              fill=sum(np.asarray(r['info']['sales']).sum() for r in rows)/demand if demand else 1.)
        if key[0].endswith('_plus'):
            counterpart = groups[key[0][:-5]+'_minus', key[1]]
            assert [r['info']['demand'] for r in rows] == [r['info']['demand'] for r in counterpart]
    for label, expected in expected_scores.items():
        for metric, value in recomputed[label, expected['scenario']].items():
            assert abs(value-expected[metric]) < 1e-12
    for row in summary['rows']:
        for metric, value in recomputed[row['label'], row['scenario']].items():
            assert abs(value-row[metric]) < 1e-12
        reference = groups['full7_5', row['scenario']]
        assert [r['info']['demand'] for r in groups[row['label'], row['scenario']]] == [r['info']['demand'] for r in reference]
    means = {label: {metric: float(np.mean([recomputed[label, k][metric] for k in range(1120000, 1120064)]))
                     for metric in ['reward_day', 'fill']} for label in summary['means']}
    for label in means:
        for metric in means[label]:
            assert abs(means[label][metric]-summary['means'][label][metric]) < 1e-12
    eligible = [a for a in ['sku', 'arrival'] if means[a]['reward_day']-means['full7_5']['reward_day'] >= .5
                and means[a]['fill']-means['full7_5']['fill'] >= -.005]
    selected = max(eligible, key=lambda a: (means[a]['reward_day'], a == 'sku')) if eligible else None
    assert eligible == summary['eligible'] and selected == summary['selected']
    assert len(groups) == 1280 and summary['counters'] == {'episodes': 1280, 'days': 35840}
    write_json(out/'verification.json', dict(status='passed', verifier_sha256=digest(__file__),
               updates_reconstructed=48, episodes_checked=1280, independent_actions_returns_gradients=True,
               paired_demand=True, selection_reconstructed=True, new_simulator_days=0))
    print('Verified pilot traces,48 score-function/Adam updates and development selection.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); verify(p.parse_args().output)
