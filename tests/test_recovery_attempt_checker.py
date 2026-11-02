import sqlite3
import pytest
from scripts.check_recovery_attempts import check
from src.experiments.recovery_execution import execute_attempt
from src.experiments.recovery_ledger import RecoveryLedger


def fixture(root):
    root.mkdir()
    plan = dict(identity='fixed', units=[dict(item='a', phase='fit', days=2,
                seconds=20., input_identity='same')],
                caps=dict(days=6, fit=60., assessment=20., total=80.), max_attempts=3)
    ledger = RecoveryLedger(root/'attempts.sqlite', **plan)
    ledger.begin('a', 'same')
    ledger.close()
    def work(b):
        b.before_step()
        return {'value': 7}
    execute_attempt(root, plan['identity'], plan['units'], plan['caps'],
                    'a', 'same', work, lambda: None)
    return plan


def test_independent_charges_and_incomplete_rejection(tmp_path):
    root = tmp_path/'valid'
    plan = fixture(root)
    result = check(root, plan, require_complete=True)
    assert result['charged']['days'] == 3
    assert result['by_status']['abandoned'] == dict(days=2, seconds=20.)
    assert result['scientific_verification'] is False


@pytest.mark.parametrize('sql', [
    "UPDATE attempts SET days=0 WHERE id=1",
    "UPDATE attempts SET seconds=0 WHERE id=1",
    "UPDATE attempts SET unit=1 WHERE id=2",
    "UPDATE attempts SET status='running' WHERE id=1",
    "UPDATE attempts SET artifact='../elsewhere' WHERE id=2",
    "UPDATE attempts SET days=2 WHERE id=2",
])
def test_rehashed_or_structural_tampering_rejected(tmp_path, sql):
    root = tmp_path/'invalid'
    plan = fixture(root)
    with sqlite3.connect(root/'attempts.sqlite') as db:
        db.execute(sql)
    with pytest.raises(ValueError):
        check(root, plan, require_complete=True)
