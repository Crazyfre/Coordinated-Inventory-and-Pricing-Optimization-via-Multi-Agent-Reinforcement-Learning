import pytest
from scripts.audit_residual_errors import statistics


def test_error_counts_units_and_probabilities():
    r=statistics([0,2,40],[1,2,38],[.25,.5,.25])
    assert r['count']==3 and r['errors']==2
    assert r['under']==1 and r['over']==1
    assert r['accuracy']==pytest.approx(1/3)
    assert r['mae_units']==5
    assert r['target_probability']==pytest.approx(1/3)
    assert r['nll']==pytest.approx(1.1552453009332422)


def test_empty_and_invalid_probability_groups():
    assert statistics([],[],[])['accuracy'] is None
    with pytest.raises(AssertionError):statistics([0],[0],[0.])
