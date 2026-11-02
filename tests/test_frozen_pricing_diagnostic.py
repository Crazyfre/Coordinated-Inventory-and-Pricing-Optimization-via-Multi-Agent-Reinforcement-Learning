import numpy as np
from scripts.diagnose_frozen_pricing import candidates, select, effects


def test_complete_interventions_and_tie_prefers_baseline():
    policies = candidates()
    assert len(policies) == 45
    for i in range(10):
        for j in range(4):
            assert policies[f"sku_{i}_{j}"][i] == j
            assert np.count_nonzero(policies[f"sku_{i}_{j}"] != 4) == 1
    rows = [dict(label=l, reward_day=1.) for l in policies]
    np.testing.assert_array_equal(select(rows), np.full(10, 4))
    rows.append(dict(label="sku_3_1", reward_day=3.))
    expected = np.full(10, 4)
    expected[3] = 1
    np.testing.assert_array_equal(select(rows), expected)


def test_paired_constant_gain():
    rows = [dict(label=l, scenario=k, reward_day=float(k + (l == "selected")),
                 fill=.6 + .01 * (l == "selected"))
            for k in range(2210000, 2210064) for l in ["baseline", "selected"]]
    result = effects(rows)
    np.testing.assert_allclose(result["reward_day"]["ci95"], [1., 1.])
    np.testing.assert_allclose(result["fill"]["ci95"], [.01, .01])
