import numpy as np
import pytest
from src.eval.review_metrics import (return_summary, success_interval,
                                    paired_shortfall, convergence, generalization_gap)


def test_zero_success_is_not_zero_uncertainty():
    result = success_interval([0]*20)
    assert result['rate'] == 0
    assert result['ci95'][1] == pytest.approx(.161125, abs=1e-6)
    assert return_summary([4])['ci95'] is None
    with pytest.raises(ValueError):
        return_summary([float('nan')])


def test_pair_by_identity_keep_negative_shortfall_and_reject_partial_pairs():
    a = [dict(seed=2, return_=7), dict(seed=1, return_=3)]
    b = [dict(seed=1, return_=5), dict(seed=2, return_=6)]
    result = paired_shortfall(a, b, 'return_')
    assert result['differences'] == [2, -1]
    assert result['cumulative'] == [2, 1]
    with pytest.raises(ValueError):
        paired_shortfall(a, b[:1], 'return_')
    with pytest.raises(ValueError):
        paired_shortfall(a+a, b, 'return_')


def test_convergence_needs_sustained_evaluation_and_reports_confirmation():
    rows = [dict(steps=10*(i+1), mean_evaluation_return=s)
            for i, s in enumerate([2, 3, 0, 2, 3, 4])]
    result = convergence(rows, 2, 3)
    assert result == dict(status='reached', onset_step=40., confirmed_step=60.)
    assert convergence([], 2)['status'] == 'unavailable'
    assert convergence(rows, 5)['status'] == 'not_reached'
    with pytest.raises(ValueError):
        convergence(rows[::-1], 2)


def test_generalization_direction_and_uncertainty():
    result = generalization_gap([10, 12, 14], [4, 6, 8])
    assert result['gap'] == 6
    assert result['ci95'][0] < 6 < result['ci95'][1]
    assert np.isfinite(result['ci95']).all()
