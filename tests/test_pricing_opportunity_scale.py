import numpy as np
import pytest
from scripts.audit_pricing_opportunity_scale import describe


def test_split_sign_selection_does_not_reuse_scoring_half():
    # Positive noisy selection flips on evaluation for the first two states.
    g=np.array([[2.]*16+[-4.]*16,[-6.]*16+[8.]*16,[3.]*32,[-1.]*32])
    r=describe(g)
    assert r['positive_mean']==2 and r['both_halves_positive']==1
    assert r['first_select_second_score']==(-4+3)/4
    assert r['second_select_first_score']==(-6+3)/4
    assert r['symmetric_transfer']==-.5
    assert r['positive_part_mean']==1
    assert describe(g*7)['symmetric_transfer']==-3.5
    with pytest.raises(ValueError):describe(g[:,:16])
