import json
import os
from pathlib import Path
import time
import pytest
from src.experiments import supervised_work as supervision
from src.experiments.execution_journal import atomic_json,digest
from src.experiments.recovery_execution import StepBudget


@pytest.mark.parametrize('failure',['malformed','read_denied'])
def test_request_admission_errors_are_preserved(tmp_path,monkeypatch,failure):
    from scripts.recovery_worker_server import run
    request=tmp_path/'request_000001.json'
    request.write_text('invalid json')
    if failure=='read_denied':
        original=Path.read_bytes
        def read(path,*args,**kwargs):
            if path==request:raise PermissionError('injected request read denial')
            return original(path,*args,**kwargs)
        monkeypatch.setattr(Path,'read_bytes',read)
    assert run(tmp_path)==1
    record=json.loads((tmp_path/'error_000001.json').read_text())
    assert record['request_sha256'] is None
    assert ('TimeoutError' if failure=='read_denied' else 'JSONDecodeError') in record['error']
    assert 'Traceback' in record['traceback']


def test_response_published_during_exit_poll_is_read_and_retained(tmp_path,monkeypatch):
    class ExitRace:
        def __init__(self,argv,cwd):self.root=Path(argv[-1]);self.closed=False
        def poll(self):
            request=self.root/'request_000001.json'
            atomic_json(self.root/'response_000001.json',dict(sequence=1,
                request_sha256=digest(request),status='failed',error='specific callback evidence'))
            return 1
        def close(self):self.closed=True
    monkeypatch.setattr(supervision,'OwnedWorker',ExitRace)
    worker=supervision.SupervisedWork(tmp_path/'scratch')
    with pytest.raises(RuntimeError,match='callback failed: specific callback evidence'):
        worker.call('unused','unused',{},StepBudget(0,time.perf_counter()+10,time.perf_counter))
    assert worker.worker is None
    assert (worker.failure_path/'request_000001.json').exists()
    assert (worker.failure_path/'response_000001.json').exists()
    assert 'specific callback evidence' in json.loads((worker.failure_path/'parent_failure.json').read_text())['error']
    worker.close()  # idempotent; diagnostics survive later cleanup
    assert worker.failure_path.exists()


@pytest.mark.skipif(os.name!='nt',reason='Windows owned worker')
@pytest.mark.parametrize('callback,expected',[('fail','fixture callback error'),
    ('invalid_json','Out of range float values'),('hard_exit','before response: 23')])
def test_real_failure_evidence_survives_cleanup(tmp_path,callback,expected):
    worker=supervision.SupervisedWork(tmp_path/'scratch')
    with pytest.raises(RuntimeError,match=expected):
        worker.call('tests.recovery_worker_callbacks',callback,{},
                    StepBudget(0,time.perf_counter()+20,time.perf_counter))
    root=worker.failure_path
    assert worker.worker is None and (root/'request_000001.json').exists()
    assert expected in json.loads((root/'parent_failure.json').read_text())['error']
    if callback!='hard_exit':
        diagnostic=json.loads((root/'error_000001.json').read_text())
        assert expected in diagnostic['error'] and 'Traceback' in diagnostic['traceback']
        assert 'payload' not in diagnostic
    if callback=='invalid_json':assert not (root/'response_000001.json').exists()
    with pytest.raises(RuntimeError,match='cannot be reused'):
        worker.call('unused','unused',{},StepBudget(0,time.perf_counter()+1,time.perf_counter))
