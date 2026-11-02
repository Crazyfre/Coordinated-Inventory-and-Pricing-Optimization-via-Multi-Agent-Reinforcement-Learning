import copy
import numpy as np
import pytest
from src.algos.cross_sku_rollout import matrix,selection,fit
from scripts.verify_cross_sku_pricing import feature_matrix,selection as independent_selection,check_ridge,check_analysis
from scripts.screen_cross_sku_pricing import assess


def test_independent_features_selection_and_both_ridge_fits():
    rng=np.random.default_rng(73);base=rng.uniform(1,5,10);obs={}
    for day in [1,8,15,22]:
        x=rng.uniform(0,50,204);x[-20:-10]=rng.uniform(1,15,10);obs[day]=x
        for arm in ['local','cross']:
            np.testing.assert_allclose(matrix(x,day,base,arm),feature_matrix(x,day,base,arm),rtol=0,atol=1e-14)
    assert [list(v) for v in selection(obs,5,base)]==independent_selection(obs,5,base)
    for arm,width in [('local',16),('cross',20)]:
        x=rng.normal(size=(256,width));x[:,12]=0;y=rng.normal(size=256);model=fit(x,y,arm)
        check_ridge(x,y,model)
        for field in ['weights','mean','scale']:
            bad=copy.deepcopy(model);bad[field][11]+=.1
            with pytest.raises(AssertionError):check_ridge(x,y,bad)


def test_independent_assessment_gates_and_tamper_rejection():
    rows=[dict(label=l,key=k,reward_day=91. if l=='cross' else 90.,fill=.7,price_counts=[0,0,0,0,40])
          for k in range(2560000,2560128) for l in ['cross','local','baseline','heuristic']]
    a=assess(rows);check_analysis(rows,a)
    assert a['primary_pass'] and a['representation_pass'] and a['heuristic_pass']
    bad=copy.deepcopy(a);bad['effects']['local:reward_day']['ci991667'][0]+=1
    with pytest.raises(AssertionError):check_analysis(rows,bad)
    with pytest.raises(AssertionError):check_analysis(rows[:-1],a)
