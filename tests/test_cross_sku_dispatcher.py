import json
import numpy as np
import pytest
from scripts import run_cross_sku_segmented as dispatcher
from scripts import cross_sku_segmented_units as units
from src.experiments.execution_journal import digest


def setup_fake(monkeypatch,tmp_path):
    source=tmp_path/'source.py';source.write_text('frozen')
    plan=['scan_000','selection']
    manifest=dict(hashes={str(source):digest(source)},items=plan,caps=dispatcher.CAPS,
                  maximum_days=dispatcher.MAXIMUM_DAYS,covers=[1]*10)
    monkeypatch.setattr(dispatcher.units,'plan',lambda:plan)
    monkeypatch.setattr(dispatcher,'new_manifest',lambda:manifest)
    calls=[]
    def perform(root,item,manifest,deadline):
        calls.append(item)
        if item=='selection':
            prior,h=dispatcher.read_committed(root,'scan_000')
            assert prior==dict(trace='synthetic') and h
            return dict(frozen=True),0
        return dict(trace='synthetic'),28
    monkeypatch.setattr(dispatcher,'perform',perform)
    return source,calls


def test_dispatcher_clean_continuation_preserves_hashes_and_zero_day_unit(tmp_path,monkeypatch):
    source,calls=setup_fake(monkeypatch,tmp_path);root=tmp_path/'run'
    a=dispatcher.run_one(root);b=dispatcher.run_one(root)
    assert calls==['scan_000','selection'] and a['total_days']==b['total_days']==28
    assert dispatcher.read_committed(root,'selection')[0]==dict(frozen=True)
    assert json.loads((root/'time.json').read_text())['completed']==calls
    with pytest.raises(ValueError,match='already'):dispatcher.run_one(root)


def test_changed_source_fails_before_scientific_callback_and_stays_failed(tmp_path,monkeypatch):
    source,calls=setup_fake(monkeypatch,tmp_path);root=tmp_path/'run'
    dispatcher.run_one(root);source.write_text('changed')
    with pytest.raises(ValueError,match='frozen source'):dispatcher.run_one(root)
    assert calls==['scan_000']
    assert json.loads((root/'time.json').read_text())['status']=='failed'
    with pytest.raises(RuntimeError):dispatcher.run_one(root)


def test_fit_unit_requires_all_samples_and_freezes_both_models():
    rng=np.random.default_rng(4);branches=[]
    for ep in range(128):
        states=[]
        for rank in [0,1]:
            local=np.r_[np.eye(10)[ep%10],rng.normal(size=6)]
            states.append(dict(sample=dict(episode=ep,rank=rank,target=float(rng.normal()),
                features=dict(local=local.tolist(),cross=np.r_[local,rng.normal(size=4)].tolist()))))
        branches.append(dict(episode=ep,states=states))
    result,days=units.fit_models(branches,np.ones(10),'selection-hash',['branch-hash']*128,{'source':'hash'})
    assert days==0 and set(result['models'])=={'local','cross'} and len(result['targets'])==256
    assert result['selection_hash']=='selection-hash' and len(result['branch_hashes'])==128
    with pytest.raises(ValueError):units.fit_models(branches[:-1],np.ones(10),'selection-hash',['x']*127,{})
