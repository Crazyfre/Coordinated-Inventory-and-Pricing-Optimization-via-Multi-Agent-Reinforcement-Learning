import numpy as np
import pytest
from scripts.diagnose_conditional_pricing import select_split,analyze


def test_future_evaluation_cannot_select_the_action():
    values=np.zeros((4,5));values[:2,0]=3;values[2:,0]=-2;values[2:,3]=100
    result=select_split(values)
    assert result['chosen']==0 and result['transferred_gain']==-2
    assert select_split(np.zeros((4,5)))['chosen']==4


def test_cluster_gate_and_incomplete_grid():
    rows=[dict(episode=e,day=d,chosen=3,transferred_gain=1.,hindsight_gain=2.)
          for e in range(16) for d in [1,8,15,22]]
    result=analyze(rows)
    assert result['promising_signal'] and result['ci95']==[1.,1.]
    with pytest.raises(ValueError):analyze(rows[:-1])
