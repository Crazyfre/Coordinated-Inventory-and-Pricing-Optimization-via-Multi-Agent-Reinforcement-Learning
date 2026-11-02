import json
import os
from pathlib import Path
import pytest
from scripts.check_recovery_attempts import check
from src.experiments.recovery_execution import execute_attempt
from src.experiments.supervised_work import SupervisedWork

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows supervised execution')
MODULE = 'tests.recovery_worker_callbacks'


def plan(seconds=10.):
    return dict(identity='supervised-fixture', units=[dict(item=str(i), phase='fit', days=2,
                seconds=seconds, input_identity=f'fixed-{i}') for i in range(2)],
                caps=dict(days=20, fit=100., assessment=10., total=110.), max_attempts=3)


def execute(root, p, index, worker, callback):
    unit = p['units'][index]
    return execute_attempt(root, p['identity'], p['units'], p['caps'], unit['item'],
                           unit['input_identity'], lambda b: worker.call(MODULE, callback, {'days':2}, b),
                           lambda: None)


def test_persistent_worker_commits_and_charges(tmp_path):
    p = plan()
    root = tmp_path/'archive'
    with SupervisedWork(tmp_path/'scratch') as worker:
        one = execute(root, p, 0, worker, 'deterministic')
        pid = worker.worker.pid
        two = execute(root, p, 1, worker, 'deterministic')
        assert worker.worker.pid == pid
        for result in (one, two):
            artifact = json.loads(Path(result['artifact']).read_text())
            assert artifact['payload'] == {'values':[0,1]}
            assert artifact['worker_receipt']['days'] == artifact['days'] == 2
    assert check(root, p, require_complete=True)['charged']['days'] == 4
    assert not list((tmp_path/'scratch').iterdir())


@pytest.mark.parametrize('callback', ['fail', 'hang', 'hard_exit'])
def test_worker_failure_is_terminal_and_fully_charged(tmp_path, callback):
    p = plan(seconds=.7 if callback == 'hang' else 10.)
    root = tmp_path/'archive'
    with SupervisedWork(tmp_path/'scratch') as worker:
        with pytest.raises((RuntimeError, TimeoutError)):
            execute(root, p, 0, worker, callback)
        assert worker.worker is None
        with pytest.raises(RuntimeError, match='terminal'):
            execute(root, p, 0, worker, 'deterministic')
    result = check(root, p)
    assert result['by_status']['failed'] == dict(days=2, seconds=p['units'][0]['seconds'])
    assert result['completed_units'] == 0
    assert result['unknown_elapsed_attempts'] == 0
    receipt = json.loads((root/'failure_000001.json').read_text())
    if callback == 'hang':
        assert receipt['observed_overrun_seconds'] > 0
        assert result['observed_failure_overrun_seconds'] == receipt['observed_overrun_seconds']


def test_rehashed_timing_receipt_tamper_rejected(tmp_path):
    import hashlib
    import sqlite3
    p = plan()
    root = tmp_path/'archive'
    with SupervisedWork(tmp_path/'scratch') as worker:
        result = execute(root, p, 0, worker, 'deterministic')
    path = Path(result['artifact'])
    artifact = json.loads(path.read_text())
    artifact['worker_receipt']['started'] = artifact['worker_receipt']['ended']+1
    path.write_text(json.dumps(artifact))
    with sqlite3.connect(root/'attempts.sqlite') as db:
        db.execute('UPDATE attempts SET sha256=? WHERE id=1', (hashlib.sha256(path.read_bytes()).hexdigest(),))
    with pytest.raises(ValueError, match='receipt'):
        check(root, p)
