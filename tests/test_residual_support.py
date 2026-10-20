import numpy as np
import pytest
from scripts.diagnose_residual_support import project_target


def test_projection_is_nearest_supported_grid_action_including_boundaries():
    center=np.array([0,40,20,20,20]);target=np.array([10,20,25,19,0])
    np.testing.assert_array_equal(project_target(center,target),[2,38,22,19,18])
    np.testing.assert_array_equal(project_target(center,center),center)
    for c in range(41):
        for t in range(41):
            actual=int(project_target(np.array([c]),np.array([t]))[0])
            possible=range(max(0,c-2),min(40,c+2)+1)
            assert abs(actual-t)==min(abs(a-t) for a in possible)
    with pytest.raises(ValueError):project_target([41],[2])
