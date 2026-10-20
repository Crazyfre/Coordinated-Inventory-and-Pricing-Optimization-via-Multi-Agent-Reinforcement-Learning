import hashlib
import json
import numpy as np
import pytest
from src.algos.learned_cover_policy import LearnedCoverPolicy
from src.algos.parameter_policy_gradient import order_action


def test_loader_preserves_actions_and_checks_identity_contract(tmp_path):
    calibration = tmp_path/'calibration.json'; calibration.write_text('{}')
    order = [str(i) for i in range(10)]
    data = dict(variant=LearnedCoverPolicy.variant, execution='deterministic_mean', series_order=order,
        cover_means=np.linspace(5, 10, 10).tolist(), calibration_sha256=hashlib.sha256(b'{}').hexdigest(),
        conditions=dict(scenario='S2', price_multiplier=1.2, pricing_scope='shared_fixed', episode_days=28,
                        order_grid=list(range(0, 201, 5)), capacity=500, billing='accepted_delivered_units'))
    path = tmp_path/'policy.json'; path.write_text(json.dumps(data))
    policy = LearnedCoverPolicy.load(path, series_order=order, calibration_path=calibration)
    obs = np.random.default_rng(9).uniform(0, 30, 260).astype(np.float32)
    np.testing.assert_array_equal(policy.act(obs, 15)[0], order_action(obs, 15, data['cover_means']))
    with pytest.raises(ValueError, match='ordering'):
        LearnedCoverPolicy.load(path, series_order=order[::-1], calibration_path=calibration)
    with pytest.raises(ValueError, match='deterministic'):
        policy.act(obs, 15, greedy=False)
    calibration.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='calibration'):
        LearnedCoverPolicy.load(path, series_order=order, calibration_path=calibration)


def test_invalid_cover_rejected():
    with pytest.raises(ValueError):
        LearnedCoverPolicy([7.5]*9, list(range(10)))
    with pytest.raises(ValueError):
        LearnedCoverPolicy([float('nan')]*10, list(range(10)))
