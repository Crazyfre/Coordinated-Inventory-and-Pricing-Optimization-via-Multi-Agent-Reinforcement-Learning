import numpy as np
from scripts.compare_averaged_targets import matrices,assess
from src.algos.rollout_correction import fit_ridge


def test_matched_targets_share_features_and_paired_baseline():
    r=dict(state=dict(obs=[5,10,15,20,1,1,3,3,400,400,5,10]),actions=[0,1],reference_action=0,sku=0,day=8,
           draws=[dict(branches=[dict(value=10),dict(value=14)]),dict(branches=[dict(value=100),dict(value=102)])])
    x,single,average,noise,mask=matrices([r],2,1)
    np.testing.assert_array_equal(single,[0,4]);np.testing.assert_array_equal(average,[0,3])
    np.testing.assert_array_equal(mask,[False,True]);np.testing.assert_array_equal(noise,[0,1])
    _,s1=fit_ridge(x,single);_,s2=fit_ridge(x,average)
    np.testing.assert_array_equal(s1,s2)


def test_assessment_excludes_noop_and_clusters_two_states_per_episode():
    records=[dict(actions=[0,1]) for _ in range(16)]
    y=np.tile([0.,2.],16);mask=np.tile([False,True],16);noise=np.zeros(32)
    p=dict(single=np.tile([999.,0.],16),averaged=y,zero=np.zeros(32))
    result=assess(records,p,y,noise,mask)
    assert result['models']['single']['episode_mse']==[4.]*8
    assert result['averaged_minus_single_mse']==-4
    assert result['difference_ci95']==[-4.,-4.]
    assert result['prediction_gate']
