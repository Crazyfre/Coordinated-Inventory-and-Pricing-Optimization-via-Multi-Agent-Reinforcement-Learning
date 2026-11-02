import copy
import numpy as np
import pytest
from scripts.diagnose_pricing_magnitude import select,analyze,TERMS
from scripts.verify_pricing_magnitude import check_analysis


def test_selection_grid_ties_and_independent_evaluation():
    v=np.zeros((128,5));v[:64,0]=10;v[64:,0]=-3;v[64:,1]=100
    assert select(np.zeros((64,5)))==4 and select(v[:64])==0
    with pytest.raises(ValueError):select(v)
    rows=[dict(episode=i,values=v.tolist(),economics={k:np.zeros((128,5)).tolist() for k in TERMS}) for i in range(20)]
    a=analyze(rows);assert a['mean_transfer']==-3 and not a['promising_signal']
    check_analysis(rows,a)
    bad=copy.deepcopy(a);bad['states'][0]['chosen']=1
    with pytest.raises(AssertionError):check_analysis(rows,bad)
    with pytest.raises(ValueError):analyze(rows[:-1])


def test_positive_signal_and_exact_budget():
    v=np.zeros((128,5));v[:,3]=2
    rows=[dict(episode=i,values=v.tolist(),economics={k:np.zeros((128,5)).tolist() for k in TERMS}) for i in range(20)]
    a=analyze(rows);check_analysis(rows,a);assert a['promising_signal'] and not a['policy_success']
    assert 20*(28+28+7+128*5*7+5*7)==91560
    keys=[int(np.random.SeedSequence([2530000,7,22,4,r,s]).generate_state(1,dtype=np.uint64)[0]) for r in range(128) for s in [0,1]]
    assert len(set(keys))==256
