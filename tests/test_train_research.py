import json
from pathlib import Path


def test_development_runner_writes_reproducible_checkpoint_and_measured_cost(tmp_path):
    from scripts.train_research import run_training
    result = run_training('joint', seed=11, updates=2, days=9, episodes=2, output=tmp_path)
    assert result['training_env_steps'] == 18
    assert result['wall_seconds'] > 0
    assert result['parameter_l2_change'] > 0
    assert result['checkpoint_roundtrip_verified'] is True
    assert len(result['evaluation']['episode_rewards']) == 2
    for name in ['checkpoint.pt','config.json','training.csv','summary.json']:
        assert (tmp_path/name).exists()
    config = json.loads((tmp_path/'config.json').read_text())
    assert config['evaluation_seed'] == 10000
    assert config['torch_threads'] == 1
