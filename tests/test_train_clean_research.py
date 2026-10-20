import json
import pytest


def test_clean_training_rejects_calibration_without_train_only_provenance(tmp_path):
    from scripts.train_clean_research import run_training
    calibration = tmp_path/'bad.json'
    calibration.write_text(json.dumps({'status':'legacy'}))
    with pytest.raises(ValueError,match='train.only'):
        run_training('mappo','S1',11,2,9,2,tmp_path/'run',calibration)


def test_clean_training_uses_random_train_windows_and_validation_only(tmp_path):
    from pathlib import Path
    from scripts.train_clean_research import run_training
    calibration = Path('outputs/evaluation_research/calibration/train_only_nb.json')
    if not calibration.exists():
        pytest.skip('Generate the frozen train-only calibration artifact first')
    result = run_training('joint','S1',11,2,9,2,tmp_path,calibration)
    assert result['training_env_steps'] == 18
    assert result['checkpoint_roundtrip_verified']
    assert result['held_out_test_used'] is False
    assert len(result['training_windows']) >= 2
    assert all(15 <= start <= 1885-9+1 for start in result['training_windows'])
    cfg = json.loads((tmp_path/'config.json').read_text())
    assert cfg['validation_start'] == 1886
    assert len(cfg['calibration_sha256']) == 64
    assert (tmp_path/'checkpoint.pt').exists()


def test_clean_r3_trains_with_fixed_price_joint_adapter(tmp_path):
    from pathlib import Path
    from scripts.train_clean_research import run_training
    calibration = Path('outputs/evaluation_research/calibration/train_only_nb.json')
    if not calibration.exists():
        pytest.skip('Generate frozen train-only calibration first')
    result = run_training('r3','S2',11,1,9,2,tmp_path,calibration)
    assert result['training_env_steps'] == 128
    assert result['checkpoint_roundtrip_verified']
    assert result['held_out_test_used'] is False


def test_clean_training_records_hash_of_consumed_calibration_bytes(tmp_path,monkeypatch):
    import hashlib
    from pathlib import Path
    from scripts.train_clean_research import run_training
    from src.algos.joint_ppo import JointPPO
    source = Path('outputs/evaluation_research/calibration/train_only_nb.json')
    if not source.exists():
        pytest.skip('Generate frozen train-only calibration first')
    original_bytes = source.read_bytes()
    calibration = tmp_path/'calibration.json'
    calibration.write_bytes(original_bytes)
    original_train = JointPPO.train
    def train_then_change_file(self,*args,**kwargs):
        result = original_train(self,*args,**kwargs)
        calibration.write_bytes(original_bytes+b'\n')
        return result
    monkeypatch.setattr(JointPPO,'train',train_then_change_file)
    run_training('joint','S1',11,1,9,1,tmp_path/'run',calibration)
    config = json.loads((tmp_path/'run'/'config.json').read_text())
    assert config['calibration_sha256'] == hashlib.sha256(original_bytes).hexdigest()
