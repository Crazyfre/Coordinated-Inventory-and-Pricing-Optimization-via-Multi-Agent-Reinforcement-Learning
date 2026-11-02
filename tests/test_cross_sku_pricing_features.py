import numpy as np
import pytest
from src.algos.cross_sku_pricing_features import features


def test_excludes_own_deficit_and_respects_observation_boundary():
    x=np.zeros(204);x[-20:-10]=2;x[0]=28;x[-10:]=100;b=np.arange(1,11)
    f=features(x,b)
    assert f[0,0]==126/500 and f[0,2]==0
    assert f[0,1]==sum(b[1:]*14)/(500*b.mean())
    assert f[0,3]==pytest.approx(14/500*126/500*.8)
    altered=x.copy();altered[10:184]=999
    np.testing.assert_array_equal(features(altered,b),f)
    # Increasing another SKU's stock reduces pressure on candidate0.
    altered=x.copy();altered[1]=14
    assert features(altered,b)[0,0]<f[0,0]
    with pytest.raises(ValueError):features(x,np.zeros(10))
