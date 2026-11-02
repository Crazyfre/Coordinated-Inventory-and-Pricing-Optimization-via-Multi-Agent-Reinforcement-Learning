import json
import pytest
from scripts.recovery_source_inventory import inventory,validate_coverage,cache_contract
from src.experiments.execution_journal import digest


def setup(root):
    files={'src/config/skus.json':json.dumps(dict(locked=True,skus=[dict(series_id='a')],protocol=dict(test_last_n_days=28))),
           'data/processed/aligned_skus.key.json':json.dumps(dict(series_ids=['a'],test_last_n_days=28)),
           'data/processed/aligned_skus.csv':'fixture', 'src/module.py':'fixed',
           'scripts/runner.py':'fixed','requirements.txt':'fixture','protocol.md':'prospective',
           'Docs/1. Research Design and Locked Configuration':'locked',
           'outputs/calibration.json':'{}','outputs/ordering.json':'{}'}
    for name,contents in files.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(contents)
    inputs={name:dict(path=f'outputs/{name}.json',sha256=digest(root/f'outputs/{name}.json'))
            for name in ('calibration','ordering')}
    return dict(inputs=inputs,source_hashes=inventory(root,inputs,'protocol.md'))


def test_required_coverage_and_new_source_detection(tmp_path):
    study=setup(tmp_path)
    checked=validate_coverage(tmp_path,study,'protocol.md')
    assert checked==study['source_hashes']
    assert 'data/processed/aligned_skus.csv' in study['source_hashes']
    (tmp_path/'src/new_module.py').write_text('new')
    with pytest.raises(ValueError,match='omits'):validate_coverage(tmp_path,study,'protocol.md')


def test_missing_mismatched_cache_and_changed_input_rejected(tmp_path):
    study=setup(tmp_path)
    (tmp_path/'outputs/ordering.json').write_text('changed')
    with pytest.raises(ValueError,match='changed'):validate_coverage(tmp_path,study,'protocol.md')
    (tmp_path/'data/processed/aligned_skus.key.json').write_text('{}')
    with pytest.raises(ValueError,match='rebuilding forbidden'):cache_contract(tmp_path)


def test_repository_cache_contract_matches_training_calibration():
    from pathlib import Path
    contract=cache_contract(Path.cwd())
    path=Path('outputs/calibration_correction/identifiable_v2/train_only_nb_identifiable_v2.json')
    calibration=json.loads(path.read_text())
    assert contract['series_ids']==calibration['series_order']
    assert contract['test_last_n_days']==28
