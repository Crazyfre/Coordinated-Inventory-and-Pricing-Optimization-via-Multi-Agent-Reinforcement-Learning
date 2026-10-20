import numpy as np
import pytest
from src.algos.arrival_aware import arrival_features,controlled_legacy_features


def obs():
    n,h=10,7;x=np.zeros(n*(1+h+14+4),dtype=np.float32)
    x[:n]=10;x[n:n+n*h].reshape(n,h)[:,0]=5
    x[-4*n:-3*n]=2;x[-2*n:-n]=400;x[-n:]=5
    return x


def test_own_timing_distinguished_but_legacy_aliases():
    a=obs();b=a.copy();pipe=b[10:80].reshape(10,7);pipe[0,0]=0;pipe[0,5]=5
    np.testing.assert_array_equal(controlled_legacy_features(a,10,7,0,1,8),controlled_legacy_features(b,10,7,0,1,8))
    assert not np.array_equal(arrival_features(a,10,7,0,1,8),arrival_features(b,10,7,0,1,8))


def test_competitor_distribution_distinguished_with_same_totals():
    a=obs();b=a.copy();b[1]+=5;b[2]-=5
    np.testing.assert_array_equal(controlled_legacy_features(a,10,7,0,1,8),controlled_legacy_features(b,10,7,0,1,8))
    assert not np.array_equal(arrival_features(a,10,7,0,1,8),arrival_features(b,10,7,0,1,8))


def test_prices_identity_time_and_dimensions():
    a=obs();base=arrival_features(a,10,7,0,1,8)
    assert len(base)==246
    b=a.copy();b[-40]=3
    assert not np.array_equal(base,arrival_features(b,10,7,0,1,8))
    assert not np.array_equal(base,arrival_features(a,10,7,1,1,8))
    assert not np.array_equal(base,arrival_features(a,10,7,0,1,15))


def test_terminal_and_noop_zero_for_both_models():
    for fn in [arrival_features,controlled_legacy_features]:
        for day,delta in [(8,0),(27,-2),(28,2)]:
            assert not fn(obs(),10,7,0,delta,day).any()
    with pytest.raises(ValueError):arrival_features(obs(),10,7,0,1,29)
