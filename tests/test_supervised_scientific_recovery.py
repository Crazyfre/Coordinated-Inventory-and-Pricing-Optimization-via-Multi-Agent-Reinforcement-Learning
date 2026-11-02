"""Real simulator/real parent-crash test; no research samples or policy fitting."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from src.experiments.execution_journal import digest
from src.experiments.recovery_execution import execute_attempt
from src.experiments.supervised_work import SupervisedWork
from scripts.check_recovery_attempts import check

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows supervised science')


def inputs():
    paths = dict(calibration='outputs/calibration_correction/identifiable_v2/train_only_nb_identifiable_v2.json',
                 ordering='outputs/rl_discovery/parameter_confirmation_v1/sku_seed211_final.json')
    return {k:dict(path=p, sha256=digest(p)) for k,p in paths.items()}


def fixture_plan():
    return dict(identity='engineering-group-recovery-only',
                units=[dict(item='group', phase='fit', days=77, seconds=120., input_identity='fixed-engineering-group')],
                caps=dict(days=231, fit=360., assessment=1., total=361.), max_attempts=3)


def run(root, scratch, crash=False):
    p = fixture_plan()
    args = dict(inputs=inputs(), key=910000301, day=22, sku=3,
                futures=[dict(replica=r, keys=[910000400+r,910000500+r]) for r in range(4)])
    def boundary(event):
        if crash and event == 'artifact_written':
            os._exit(23)
    with SupervisedWork(scratch) as worker:
        return execute_attempt(root, p['identity'], p['units'], p['caps'], 'group',
            'fixed-engineering-group', lambda b: worker.call('scripts.recovery_scientific_units',
                'supervised_group', args, b), lambda:None, boundary=boundary)


def test_group_payload_identical_after_real_parent_crash(tmp_path):
    clean = run(tmp_path/'clean', tmp_path/'scratch_clean')
    root = tmp_path/'recovered'
    code = 'from tests.test_supervised_scientific_recovery import run; import sys; run(sys.argv[1],sys.argv[2],True)'
    process = subprocess.run([sys.executable, '-c', code, str(root), str(tmp_path/'scratch_crash')],
                             capture_output=True, text=True, timeout=180,
                             creationflags=subprocess.CREATE_NO_WINDOW)
    assert process.returncode == 23, process.stderr
    recovered = run(root, tmp_path/'scratch_recovered')
    original = json.loads(Path(clean['artifact']).read_text())
    abandoned = json.loads((root/'attempt_000001.json').read_text())
    replayed = json.loads(Path(recovered['artifact']).read_text())
    # Execution receipts differ. Every scientific field/float/trace must match.
    assert original['payload'] == abandoned['payload'] == replayed['payload']
    assert original['days'] == abandoned['days'] == replayed['days'] == 77
    report = check(root, fixture_plan(), require_complete=True)
    assert report['charged']['days'] == 154
    assert report['by_status']['abandoned'] == dict(days=77, seconds=120.)
    assert report['completed_units'] == 1 and report['attempts'] == 2
    assert report['unknown_elapsed_attempts'] == 1
