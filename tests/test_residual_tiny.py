import numpy as np
import pytest
from scripts.diagnose_residual_tiny import select_rows,summarize


def test_selection_balanced_ordered_and_excludes_interior():
    c=np.array([[0],[2],[1],[0],[0],[1],[1]])
    y=np.array([[0],[3],[2],[2],[1],[1],[3]])
    assert select_rows(c,y,0,k=2).tolist()==[0,5,2,4,3,6]
    with pytest.raises(ValueError):select_rows(c,y,0,k=3)


def test_per_class_fidelity_and_mean_probability():
    p=np.array([[.9,.05,.05],[.1,.8,.1],[.2,.4,.4]])
    r=summarize(p,np.array([0,1,2]),np.array([0,1,2]))
    assert r['accuracy']==pytest.approx(2/3)
    assert r['target_probability']==pytest.approx(.7)
    assert r['classes']['2']['accuracy']==0
