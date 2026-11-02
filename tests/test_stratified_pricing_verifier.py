import copy
import numpy as np
import pytest
from src.algos.stratified_pricing import features,select_candidates,fit
from scripts.screen_stratified_pricing import assess
from scripts.verify_stratified_pricing import feature_matrix,selection,check_ridge,check_analysis


def test_independent_features_selection_and_ridge():
    rng=np.random.default_rng(66);base=rng.uniform(1,20,10);matrix=[]
    for ep in range(128):
        obs={d:rng.uniform(0,20,204).astype(np.float32) for d in [1,8,15,22]}
        assert selection(obs,ep)==[list(v) for v in select_candidates(obs,ep)]
        for d,sku in selection(obs,ep):
            actual=features(obs[d],d,base);expected=feature_matrix(obs[d],d,base)
            np.testing.assert_array_equal(actual,expected);matrix.append(actual[sku])
    x=np.array(matrix);y=rng.normal(size=256);w=fit(x,y);check_ridge(x,y,w)
    bad=w.copy();bad[0]+=.01
    with pytest.raises(AssertionError):check_ridge(x,y,bad)


def test_independent_nonconstant_intervals_and_gate_tampering():
    rows=[]
    for key in range(2520000,2520064):
        for label in ['learned','baseline','heuristic']:
            rows.append(dict(label=label,key=key,reward_day=100.+(key%7)/3+(1. if label=='learned' else 0),
                fill=.7+(.001 if label=='learned' else 0),price_counts=[0,0,0,0,40]))
    a=assess(rows);check_analysis(rows,a)
    bad=copy.deepcopy(a);bad['effects']['baseline:reward_day']['ci9875'][0]+=.001
    with pytest.raises(AssertionError):check_analysis(rows,bad)
    bad=copy.deepcopy(a);bad['primary_pass']=False
    with pytest.raises(AssertionError):check_analysis(rows,bad)
