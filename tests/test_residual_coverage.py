import numpy as np
from scripts.audit_residual_coverage import neighbors,transition_counts


def test_transitions_partition_and_net_accuracy_change():
    r=transition_counts(np.array([0,0,0,0]),np.array([0,1,0,1]),np.array([0,0,1,1]))
    assert r==dict(count=4,retained_correct=1,corrected=1,regressed=1,persistent_error=1)


def test_neighbors_exclude_whole_episode_and_handle_constant_coordinate():
    fit=np.array([[0.,0],[.01,0],[10.,0],[11.,0]])
    r=neighbors(fit,np.array([[.005,0]]),episode_days=2)
    assert r['fit_index'].tolist()==[2,2,1,1]
    assert r['holdout_index'][0]==0
    assert r['scale'][1]==1
    assert np.isfinite(r['fit_distance']).all()
