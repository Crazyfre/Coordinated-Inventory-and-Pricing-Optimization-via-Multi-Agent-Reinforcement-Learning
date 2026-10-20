import numpy as np
import pytest
import torch
from src.algos.parameter_policy_gradient import initial_mean, sigmas, order_action, antithetic_gradient, ParameterPolicyGradient
from src.algos.imitation_ordering import teacher_indices


def test_baseline_and_support_exact():
    rng = np.random.default_rng(8)
    for _ in range(80):
        x = rng.uniform(0, 40, 260).astype(np.float32)
        x[:80] = np.rint(x[:80])
        expected = teacher_indices(torch.tensor(x), 10, 7, cover=7.5).numpy()
        for arm in ['sku', 'arrival']:
            np.testing.assert_array_equal(order_action(x, 8, initial_mean(arm)), expected)
    assert np.all(order_action(np.ones(260, np.float32)*10000, 28, np.ones(21)*1e4) <= 40)


def test_score_gradient_linear_return_identity():
    e = np.r_[np.eye(10), -np.eye(10)]*np.sqrt(10)
    sigma = sigmas('sku'); slope = np.arange(10)/10
    differences = 2*(e*sigma)@slope
    np.testing.assert_allclose(antithetic_gradient(e, differences, sigma), slope, atol=1e-14)
    np.testing.assert_array_equal(antithetic_gradient(e, np.zeros(20), sigma), np.zeros(10))


def test_arrival_context_changes_state_policy():
    x = np.zeros(260, np.float32); x[-10:] = 10
    x[10:80].reshape(10, 7)[:, 6] = 5
    y = x.copy(); y[10:80].reshape(10, 7)[:] = 0; y[10:80].reshape(10, 7)[:, 0] = 5
    p = initial_mean('arrival'); p[10:20] = 4
    assert np.any(order_action(x, 8, p) != order_action(y, 8, p))


def test_update_reproducibility_and_bounds():
    a, b = ParameterPolicyGradient('arrival', 5), ParameterPolicyGradient('arrival', 5)
    for _ in range(5):
        e = a.rng.normal(size=(8, 21)); np.testing.assert_array_equal(e, b.rng.normal(size=(8, 21)))
        a.update(e, np.arange(8)); b.update(e, np.arange(8))
    assert a.state() == b.state() and a.steps == 5
    assert np.all((a.mean[:10] >= 0) & (a.mean[:10] <= 16))
    with pytest.raises(ValueError):
        antithetic_gradient(np.ones((8, 10)), np.ones(8), np.zeros(10))
