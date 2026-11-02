import copy
import numpy as np
import pytest
from scripts.confirm_pricing_candidates import CANDIDATES,TERMS,analyze
from scripts.verify_pricing_candidates import check_statistics


def rows():
    return [dict(candidate=list(c),differences=(np.linspace(-1,1,128)+j-1).tolist(),
        fill_differences=[0.]*128,economics={k:[0.]*128 for k in TERMS}) for j,c in enumerate(CANDIDATES)]


def test_independent_intervals_and_tamper_rejection():
    r=rows();a=analyze(r);check_statistics(r,a)
    assert a['positive_states']==2 and a['policy_success'] is False
    bad=copy.deepcopy(a);bad['states'][0]['ci9875'][0]+=.01
    with pytest.raises(AssertionError):check_statistics(r,bad)


def test_no_candidate_or_future_selection():
    r=rows()
    with pytest.raises(ValueError):analyze(r[:3])
    r[0]['differences'].pop()
    with pytest.raises(ValueError):analyze(r)
