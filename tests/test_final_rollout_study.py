import numpy as np
from scripts.run_final_rollout_study import prediction_gates


def test_beating_single_only_does_not_open_validation():
    result=dict(models={k:dict(episode_mse=[v]*64) for k,v in [('single',20.),('averaged',5.),('zero',2.)]})
    gates=prediction_gates(result)
    assert gates['comparisons']['single']['passed']
    assert not gates['comparisons']['zero']['passed']
    assert not gates['passed']


def test_both_comparisons_require_strict_upper_bound_below_zero():
    result=dict(models={k:dict(episode_mse=[v]*64) for k,v in [('single',20.),('averaged',1.),('zero',2.)]})
    assert prediction_gates(result)['passed']
    result['models']['averaged']['episode_mse']=[2.]*64
    assert not prediction_gates(result)['passed']


def test_uncertain_negative_mean_does_not_open_validation():
    delta=np.tile([-100.,99.],32)
    result=dict(models={'single':dict(episode_mse=[200.]*64),'zero':dict(episode_mse=[200.]*64),
                        'averaged':dict(episode_mse=(200+delta).tolist())})
    gates=prediction_gates(result)
    assert gates['comparisons']['zero']['delta']<0
    assert not gates['passed']
