import copy
import numpy as np
import pytest
import torch
from scripts.assess_compact_residual import analyze, action_generator, KEYS, SEEDS


def fixture_rows(gain=2., fill=0.):
    rows = []
    labels = ['projected', 'full7.5', 'center', 'constant', 'ss']
    labels += [f'{m}_{s}' for s in SEEDS for m in ['sampled', 'greedy', 'initial']]
    for k in KEYS:
        for label in labels:
            change = label.startswith('sampled_')
            rows.append(dict(label=label, scenario=k, reward_day=100.+gain*change, fill=.6+fill*change))
    return rows


def test_constant_effect_gate_and_service_failure():
    result = analyze(fixture_rows(), replicates=100)
    assert result['learning_gate']
    assert result['endpoints']['initial_reward_day']['ci9875'] == [2., 2.]
    assert not analyze(fixture_rows(gain=0), replicates=100)['learning_gate']
    assert not analyze(fixture_rows(fill=-.006), replicates=100)['learning_gate']


def test_pairing_missing_duplicate_and_bad_seed():
    rows = fixture_rows()
    with pytest.raises(ValueError):
        analyze(rows[:-1])
    rows[-1] = rows[0]
    with pytest.raises(ValueError):
        analyze(rows)
    rows = fixture_rows()
    for r in rows:
        if r['label'] == 'sampled_33':
            r['reward_day'] = 99.
    assert not analyze(rows, replicates=100)['every_seed_reward_positive']


def test_crossed_axes_preserve_scenario_pairing_and_seed_variation():
    rows = fixture_rows()
    for row in rows:
        row['reward_day'] += (row['scenario']-KEYS[0])*100
    base = analyze(rows, replicates=400)
    assert base['endpoints']['projected_reward_day']['ci9875'] == [2., 2.]
    changed = copy.deepcopy(rows)
    for r in changed:
        if r['label'].startswith('sampled_'):
            r['reward_day'] += {'sampled_11': -1, 'sampled_22': 0, 'sampled_33': 1}[r['label']]
    result = analyze(changed, replicates=400)
    assert result['endpoints']['projected_reward_day']['seed_means'] == [1., 2., 3.]
    assert result['endpoints']['projected_reward_day']['ci9875'][0] < 2
    assert result == analyze(changed, replicates=400)


def test_policy_rng_is_separate_and_reproducible():
    torch.manual_seed(17); before = torch.get_rng_state().clone()
    a = torch.rand(20, generator=action_generator(11, KEYS[0]))
    b = torch.rand(20, generator=action_generator(11, KEYS[0]))
    c = torch.rand(20, generator=action_generator(22, KEYS[0]))
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert torch.equal(before, torch.get_rng_state())
