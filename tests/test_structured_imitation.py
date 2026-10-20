import numpy as np
import pytest
from src.algos.structured_imitation import position_features,fit_intervals,InventoryPositionLinearPolicy


def test_features_and_mapping_use_pipeline_with_no_teacher_cover():
    obs=np.array([[10,15,5,0,10],[10,0,0,0,10]],dtype=np.float32)
    np.testing.assert_array_equal(position_features(obs,1,2),[[[2,6,1]],[[2,2,1]]])
    policy=InventoryPositionLinearPolicy(1,2,[2,-1,.2])
    assert policy.predict(obs).tolist()==[[0],[3]]
    with pytest.raises(ValueError,match='deterministic'):
        policy.act(obs[0],greedy=False)


def test_interval_fit_learns_from_labels_including_censored_endpoints():
    rng=np.random.default_rng(4)
    features=np.column_stack([rng.uniform(0,20,1000),rng.uniform(0,40,1000),np.ones(1000)])
    labels=np.ceil(np.maximum(features@np.array([3.,-1.,.1]),0)).clip(0,40).astype(int)
    fitted=fit_intervals(features,labels)
    pred=np.ceil(np.maximum(features@fitted['coefficients'],0)).clip(0,40)
    np.testing.assert_array_equal(pred,labels)
    assert abs(fitted['coefficients'][0]-3)<.02
    assert fitted['labels']==1000


def test_checkpoint_schema_rejects_wrong_actor_and_roundtrips(tmp_path):
    path=tmp_path/'policy.json'
    policy=InventoryPositionLinearPolicy(1,2,[2,-1,.2]);policy.save(path)
    restored=InventoryPositionLinearPolicy.load(path,1,2)
    np.testing.assert_array_equal(policy.coefficients,restored.coefficients)
    with pytest.raises(ValueError,match='schema'):
        InventoryPositionLinearPolicy.load(path,2,2)
    with pytest.raises(ValueError):
        fit_intervals([[1,2,1]],[41])
