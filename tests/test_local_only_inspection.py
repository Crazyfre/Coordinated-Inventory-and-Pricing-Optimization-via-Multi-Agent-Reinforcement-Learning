import numpy as np
import pytest
from scripts.inspect_local_only import describe


def test_confidence_deficit_and_error_partition():
    r=describe(np.array([0,1,2]),np.array([0,2,2]),np.array([.8,.05,.995]),np.array([.8,.9,.995]))
    assert r['errors']==1 and r['correct_below_99']==1 and r['high_confidence_errors']==1
    assert r['expected_sampled_errors']==pytest.approx(1.155)
    assert r['over']==1 and r['under']==0


def test_empty_group_is_not_perfect_accuracy():
    z=np.array([]);r=describe(z,z,z,z)
    assert r['accuracy'] is None and r['mean_target_probability'] is None and r['expected_sampled_errors']==0
