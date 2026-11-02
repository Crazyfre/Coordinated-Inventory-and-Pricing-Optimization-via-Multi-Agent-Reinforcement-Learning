import json
import pytest
from scripts.recovery_campaign_plan import build,bind_request
from scripts.recovery_campaign_dispatch import validate_request,load_dependency
from src.experiments.execution_journal import digest
from tests.test_recovery_campaign_plan import study


def test_source_callback_and_dependency_bindings(tmp_path):
    source = tmp_path/'source.py'
    source.write_text('fixed')
    config = study()
    config['callback'] = dict(module='scripts.recovery_campaign_dispatch',function='dispatch')
    config['source_hashes'] = {str(source):digest(source)}
    units = build(config)['units']
    first = units[0]
    request = bind_request(config,first,[])
    assert validate_request(request)[1] == first
    source.write_text('changed')
    with pytest.raises(ValueError,match='source'):validate_request(request)
    source.write_text('fixed')
    first['days'] = 0
    with pytest.raises(ValueError,match='plan'):bind_request(config,first,[])


def test_dependency_artifact_identity_and_hash(tmp_path):
    path = tmp_path/'artifact.json'
    path.write_text(json.dumps(dict(item='scan_000',input_identity='frozen',payload={'episode':0})))
    record = dict(item='scan_000',input_identity='frozen',path=str(path),sha256=digest(path))
    assert load_dependency(record) == {'episode':0}
    record['input_identity'] = 'changed'
    with pytest.raises(ValueError,match='identity'):load_dependency(record)
    record['input_identity'] = 'frozen'
    path.write_text('{}')
    with pytest.raises(ValueError,match='hash'):load_dependency(record)
