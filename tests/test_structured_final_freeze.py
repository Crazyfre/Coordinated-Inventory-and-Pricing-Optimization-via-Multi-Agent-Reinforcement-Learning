import json
import numpy as np
import pytest
from scripts import evaluate_structured_final as runner
from scripts.verify_structured_final import check_effect
from scripts.verify_structured_final import check_baseline
from src.eval.structured_final import effect


def test_freeze_requires_verified_recovery_and_unchanged_final_weights(tmp_path,monkeypatch):
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
    runs=[]
    for condition in ['main','pooled','independent']:
        for seed in runner.SEEDS:
            path=tmp_path/f'{condition}_{seed}'/'final.json'
            write(path,dict(steps=64,mean=[7.5]*(1 if condition=='pooled' else 10)))
            runs.append(dict(condition=condition,seed=seed,checkpoint_sha256=runner.digest(path)))
    write(tmp_path/'manifest.json',dict(hashes={},existing_main={}))
    write(tmp_path/'summary.json',dict(status='complete',runs=runs))
    write(tmp_path/'verification.json',dict(status='passed'))
    with pytest.raises(ValueError,match='recovery'):runner.freeze(tmp_path)
    write(tmp_path/'verification.json',dict(status='passed',recovery_accounting_verified=True))
    manifest,policies=runner.freeze(tmp_path)
    assert len(policies)==30 and manifest['start']==1914
    np.testing.assert_array_equal(policies['pooled_211'],np.full(10,7.5))
    write(tmp_path/'main_211'/'final.json',dict(steps=64,mean=[8]*10))
    with pytest.raises(ValueError,match='checkpoint changed'):runner.freeze(tmp_path)


def test_independent_interval_checker_rejects_modified_interval():
    values=np.arange(500,dtype=float).reshape(10,50)/500
    endpoint=effect(values,.975,20261023)
    check_effect(values,endpoint,.975,20261023)
    endpoint['interval'][0]+=.01
    with pytest.raises(AssertionError):check_effect(values,endpoint,.975,20261023)


def test_baseline_checker_catches_wrong_actions_and_tracks_rounding_boundary():
    x=np.zeros(260,np.float32);x[-10:]=10
    for label,action in [('constant',2),('ss',16),('projected',12)]:
        row=dict(label=label,obs=x.tolist(),actions=[action]*10)
        assert check_baseline(row,np.array([5.,-1.,0.]))==0
        row['actions']=[action+3]*10
        with pytest.raises(AssertionError):check_baseline(row,np.array([5.,-1.,0.]))
    x[-10:]=2.5
    row=dict(label='constant',obs=x.tolist(),actions=[0]*10)
    assert check_baseline(row,np.zeros(3))==10
    row['actions']=[1]*10
    assert check_baseline(row,np.zeros(3))==10
    row['actions']=[2]*10
    with pytest.raises(AssertionError):check_baseline(row,np.zeros(3))
