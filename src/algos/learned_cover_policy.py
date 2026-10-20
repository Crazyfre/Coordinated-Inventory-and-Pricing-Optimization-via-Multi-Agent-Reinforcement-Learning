"""Read-only deployment interface for verified parameter-gradient research policies.

Loading is not permission to replace the demo or to claim MAPPO coordination.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from src.algos.parameter_policy_gradient import order_action


class LearnedCoverPolicy:
    variant = 'verified_parameter_gradient_cover_v1'

    def __init__(self, parameters, series_order):
        self.parameters = np.asarray(parameters, float)
        self.series_order = tuple(series_order)
        if self.parameters.shape != (10,) or not np.isfinite(self.parameters).all() or np.any((self.parameters < 0) | (self.parameters > 16)):
            raise ValueError('ten finite learned cover means in[0,16] required')
        if len(self.series_order) != 10 or len(set(self.series_order)) != 10:
            raise ValueError('ten unique ordered series identifiers required')

    def act(self, obs, day, greedy=True):
        if not greedy:
            raise ValueError('exported controller is deterministic mean execution only')
        return order_action(obs, day, self.parameters), 0.

    @classmethod
    def load(cls, path, *, series_order, calibration_path):
        path = Path(path); data = json.loads(path.read_text())
        if data.get('variant') != cls.variant or data.get('execution') != 'deterministic_mean':
            raise ValueError('matching verified parameter-gradient export schema required')
        if list(series_order) != data['series_order']:
            raise ValueError('SKU ordering differs from trained policy')
        calibration_hash = hashlib.sha256(Path(calibration_path).read_bytes()).hexdigest()
        if calibration_hash != data['calibration_sha256']:
            raise ValueError('calibration differs from verified study')
        if data['conditions'] != dict(scenario='S2', price_multiplier=1.2, pricing_scope='shared_fixed',
                                      episode_days=28, order_grid=list(range(0, 201, 5)), capacity=500,
                                      billing='accepted_delivered_units'):
            raise ValueError('unsupported study contract')
        return cls(data['cover_means'], series_order)
