import json
from pathlib import Path
import pytest
from scripts.recovery_campaign_plan import build
from scripts.recovery_campaign_runner import execute_unit,committed_dependencies
from src.experiments.execution_journal import digest
from tests.test_recovery_campaign_plan import study


class FakeWorker:
    def __init__(self):self.requests=[]
    def call(self,module,function,request,budget):
        self.requests.append(request)
        for _ in range(request['unit']['days']):budget.before_step()
        return dict(item=request['unit']['item'])


def config(tmp_path):
    source = tmp_path/'frozen'
    source.write_text('fixture')
    value = study()
    value['callback'] = dict(module='scripts.recovery_campaign_dispatch',function='dispatch')
    value['source_hashes'] = {str(source):digest(source)}
    value['inputs'] = {name:dict(path=str(source),sha256=digest(source)) for name in ('ordering','calibration')}
    return value,source


def test_adapter_uses_committed_dependencies_and_prevents_skip(tmp_path):
    value,source = config(tmp_path)
    root = tmp_path/'archive'
    worker = FakeWorker()
    one = execute_unit(root,value,0,worker)
    assert one['days']==28 and worker.requests[0]['dependencies']==[]
    with pytest.raises(ValueError,match='next'):execute_unit(root,value,2,worker)
    two = execute_unit(root,value,1,worker)
    assert two['charged']['days']==56
    plan = build(value)
    # Direct reader test only: no concurrent execution in this fixture.
    unit = dict(specification=dict(dependencies=['scan_000','scan_001']))
    records = committed_dependencies(root,plan,unit)
    assert [r['item'] for r in records]==['scan_000','scan_001']
    assert records[0]['sha256']==digest(one['artifact'])
    assert records[1]['input_identity']==plan['units'][1]['input_identity']
    with pytest.raises(ValueError,match='not committed'):
        committed_dependencies(root,plan,dict(specification=dict(dependencies=['scan_002'])))
    Path(one['artifact']).write_text('changed between invocations')
    fresh=FakeWorker()
    with pytest.raises(ValueError,match='corruption'):
        execute_unit(root,value,2,fresh)
    assert fresh.requests==[]


def test_source_change_fails_reserved_attempt_before_worker(tmp_path):
    value,source = config(tmp_path)
    root = tmp_path/'archive'
    worker = FakeWorker()
    source.write_text('changed')
    with pytest.raises(ValueError,match='source'):execute_unit(root,value,0,worker)
    assert worker.requests==[]
    source.write_text('fixture')
    with pytest.raises(RuntimeError,match='terminal'):execute_unit(root,value,0,worker)
    assert json.loads((root/'failure_000001.json').read_text())['attempt']==1


def test_launch_preflight_is_required_and_runtime_mismatch_terminal(tmp_path,monkeypatch):
    from scripts.recovery_campaign_runner import execute_launch_unit
    from scripts import recovery_source_inventory as sources
    value,_=config(tmp_path)
    seen=[]
    monkeypatch.setattr(sources,'validate_coverage',lambda root,study,protocol:seen.append(protocol))
    monkeypatch.setattr(sources,'runtime_identity',lambda:dict(version='actual'))
    worker=FakeWorker()
    with pytest.raises(ValueError,match='runtime'):
        execute_launch_unit(tmp_path/'archive',value,0,worker,'frozen.md',dict(version='different'))
    assert seen==['frozen.md'] and worker.requests==[]
    assert (tmp_path/'archive'/'failure_000001.json').is_file()


def test_batch_end_closes_worker_before_commit_and_clears_fullcheck_marker(tmp_path):
    value,_=config(tmp_path)
    worker=FakeWorker();closed=[]
    worker.close=lambda:closed.append(True)
    execute_unit(tmp_path/'archive',value,0,worker,close_worker=True)
    assert closed==[True] and worker.validated_campaign is None


def test_preflight_hashes_reused_only_in_current_unit_extras_still_checked(tmp_path,monkeypatch):
    from scripts import recovery_campaign_runner as runner
    value,source=config(tmp_path)
    extra=tmp_path/'extra';extra.write_text('fixed')
    value['source_hashes'][str(extra)]=digest(extra)
    reads=[];real=runner.digest
    def tracked(path):reads.append(str(path));return real(path)
    monkeypatch.setattr(runner,'digest',tracked)
    def preflight():
        checksum=tracked(source)
        if checksum!=value['source_hashes'][str(source)]:raise ValueError('changed preflight source')
        return {str(source):checksum}
    worker=FakeWorker();root=tmp_path/'archive'
    runner.execute_unit(root,value,0,worker,preflight=preflight)
    # One preflight read plus the two independently bound input reads, no
    # duplicate source pass; additional manifest paths are still hashed.
    assert reads.count(str(source))==3 and str(extra) in reads
    source.write_text('changed')
    with pytest.raises(ValueError,match='preflight source'):
        runner.execute_unit(root,value,1,worker,preflight=preflight)
    assert len(worker.requests)==1
