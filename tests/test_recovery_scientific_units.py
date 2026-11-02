"""Simulator engineering validation only; these seeds are not research samples."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from scripts.audit_repeated_futures import snapshot, resample_future
from scripts.diagnose_conditional_pricing import advance, suffix, STUDY
from scripts.diagnose_counterfactual_rollouts import clone_environment
from scripts.screen_cross_sku_pricing import reset
from scripts.train_compact_residual import INPUTS
from scripts.recovery_scientific_units import paired_future
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env
from src.experiments.recovery_execution import StepBudget


def test_real_simulator_prefix_reconstruction_and_paired_traces():
    torch.set_num_threads(1)
    covers = np.array(json.loads((STUDY/'sku_seed211_final.json').read_text())['mean'])
    params = load_identifiable_training_calibration(INPUTS['calibration'])
    def make():
        return make_research_env('S2', split='train', start=15,
                                 params=params, pricing_scope='per_sku')
    env = make()
    key = 910000001  # Engineering-only namespace, excluded from any fit/evaluation.
    reset(env, key)
    deadline = time.perf_counter()+120
    counters = {'days': 0}
    states = {}
    original = []
    for day in range(1, 29):
        if day in (1, 8, 15, 22):
            states[day] = clone_environment(env)
        row, _ = advance(env, covers, 1, 0, 4, counters, deadline)
        original.append(row)
    reconstructed_days = 0
    for day, state in states.items():
        sku = (day-1)//7
        keys = [910000100+day, 910000200+day]
        maximum = (day-1)+2*(29-day)
        budget = StepBudget(maximum, deadline, time.perf_counter)
        result = paired_future(make(), covers, key, day, sku, keys, budget,
                               expected_before=snapshot(state))
        assert result['prefix'] == original[:day-1]
        for branch in result['branches']:
            direct = resample_future(state, keys)
            assert branch['initial'] == snapshot(direct)
            expected = suffix(direct, covers, sku, branch['action'], counters, deadline)
            assert branch['trace'] == expected
            assert branch['value'] == sum(.99**i*r['reward'] for i, r in enumerate(expected))
        assert budget.days == maximum
        reconstructed_days += budget.days
    assert reconstructed_days == 182
    assert counters['days'] == 168
    assert reconstructed_days+counters['days'] == 350
