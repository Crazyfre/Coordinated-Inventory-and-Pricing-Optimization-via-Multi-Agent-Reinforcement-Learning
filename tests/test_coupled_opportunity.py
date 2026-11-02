import numpy as np
from scripts.diagnose_coupled_opportunity import prices, actions


def test_discount_threshold_strict_and_no_global_state():
    obs = np.zeros(187, np.float32)
    obs[-20:-10] = 2
    obs[:10] = 21
    np.testing.assert_array_equal(prices(obs, 22, 1), np.full(10, 4))
    obs[3] = 22
    expected = np.full(10, 4)
    expected[3] = 3
    np.testing.assert_array_equal(prices(obs, 22, 1), expected)
    expected[3] = 0
    np.testing.assert_array_equal(prices(obs, 22, 2), expected)


def test_ordering_variants_preserve_grid_and_horizon():
    obs = np.zeros(260, np.float32)
    obs[-10:] = 10
    covers = np.full(10, 7.5)
    np.testing.assert_array_equal(actions(obs, 22, covers, 0), np.full(10, 15))
    np.testing.assert_array_equal(actions(obs, 22, covers, 1), np.full(10, 6))
    np.testing.assert_array_equal(actions(obs, 22, covers, 2), np.zeros(10))
    np.testing.assert_array_equal(actions(obs, 25, covers, 1), np.zeros(10))
