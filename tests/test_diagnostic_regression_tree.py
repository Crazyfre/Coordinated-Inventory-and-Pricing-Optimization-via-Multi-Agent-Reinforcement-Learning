import copy
import numpy as np
import pytest
from src.algos.diagnostic_regression_tree import fit_tree,predict_tree
from scripts.verify_nonlinear_pricing import check_node
from scripts.diagnose_nonlinear_pricing import metrics


def test_exact_split_ties_and_independent_tamper_check():
    x=np.repeat(np.arange(4),8).astype(float);x=np.column_stack([x,x])
    y=np.repeat([-3.,-3.,2.,2.],8)
    tree=fit_tree(x,y)
    assert tree['feature']==0 and tree['threshold']==1.5
    np.testing.assert_array_equal(predict_tree(tree,x),y)
    assert check_node(tree,x,y)==1
    bad=copy.deepcopy(tree);bad['left']['mean']+=1
    with pytest.raises(AssertionError):check_node(bad,x,y)


def test_minimum_support_and_no_positive_prediction_gate():
    x=np.arange(15).reshape(-1,1);y=np.arange(15)
    tree=fit_tree(x,y);assert 'feature' not in tree
    m=metrics(np.full(256,-1.),np.ones(256),np.ones((256,2)))
    assert not m['continuation_flag'] and m['half_means']==[None,None]
    # Eight predictions concentrated in four windows cannot pass.
    pred=np.full(256,-1.);pred[:8]=1
    assert not metrics(pred,np.ones(256),np.ones((256,2)))['continuation_flag']
    pred[:]=-1;pred[::2]=1
    halves=np.ones((256,2));halves[:,1]=-1
    assert not metrics(pred,np.ones(256),halves)['continuation_flag']
