import numpy as np
import pytest
import torch
from scripts.diagnose_residual_budget import batches


def test_seeded_epochs_and_exact_partial_budget():
    actual=list(batches(10,8,batch_size=4))
    assert len(actual)==8 and [len(b) for b in actual]==[4,4,2,4,4,2,4,4]
    g=torch.Generator().manual_seed(11)
    for epoch in range(2):
        expected=torch.randperm(10,generator=g)
        np.testing.assert_array_equal(torch.cat(actual[epoch*3:epoch*3+3]),expected)
    assert all(torch.equal(a,b) for a,b in zip(actual,batches(10,8,4)))
    with pytest.raises(ValueError):list(batches(0,1))
