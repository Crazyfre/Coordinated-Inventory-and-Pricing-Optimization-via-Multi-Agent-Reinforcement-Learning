import numpy as np
from scripts.confirm_parameter_rl import analyze, SEEDS, KEYS


def test_five_seed_pairing_and_zero_gain():
    labels = [f'{mode}_{s}' for s in SEEDS for mode in ['learned', 'initial']]+['full7_5', 'projected', 'constant', 'ss']
    rows = [dict(label=label, scenario=k, reward_day=(k-KEYS[0])*10.+(2 if label.startswith('learned') else 0), fill=.65)
            for k in KEYS for label in labels]
    result = analyze(rows)
    assert result['learning_gate']
    assert result['endpoints']['full7_5_reward_day']['ci9875'] == [2., 2.]
    for row in rows:
        if row['label'].startswith('learned'):
            row['reward_day'] -= 2
    assert not analyze(rows)['learning_gate']


def test_bad_seed_and_service_are_not_hidden_by_other_runs():
    labels = [f'{mode}_{s}' for s in SEEDS for mode in ['learned', 'initial']]+['full7_5', 'projected', 'constant', 'ss']
    rows = [dict(label=l, scenario=k, reward_day=100.+(3 if l.startswith('learned') else 0), fill=.65)
            for k in KEYS for l in labels]
    for row in rows:
        if row['label'] == 'learned_255':
            row['reward_day'] = 99.
    assert not analyze(rows)['every_seed_reward_positive']
    for row in rows:
        if row['label'].startswith('learned'):
            row['reward_day'] = 103.; row['fill'] = .64
    assert not analyze(rows)['learning_gate']
