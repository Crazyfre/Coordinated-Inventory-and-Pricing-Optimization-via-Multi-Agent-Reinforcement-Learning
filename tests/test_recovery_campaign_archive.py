import hashlib
import json
import sqlite3
import time
import pytest
from scripts.check_recovery_campaign_archive import check
from scripts.recovery_campaign_plan import build
from scripts.recovery_campaign_runner import execute_unit
from src.experiments.execution_journal import digest
from tests.test_recovery_campaign_runner import config


class ReceiptWorker:
    def call(self,module,function,arguments,budget):
        started=time.perf_counter()
        request=dict(sequence=1,module=module,function=function,arguments=arguments,
                     maximum_days=budget.maximum,deadline=budget.deadline,dispatched=started)
        for _ in range(28):budget.before_step()
        ended=time.perf_counter()
        checksum=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
        budget.worker_receipt=dict(sequence=1,request_sha256=checksum,days=28,
                                   started=started,ended=ended,received=ended,request=request)
        return dict(engineering_fixture=True)


def test_prefix_binding_checker_and_rehashed_request_tampering(tmp_path):
    study,_=config(tmp_path)
    root=tmp_path/'archive'
    result=execute_unit(root,study,0,ReceiptWorker())
    plan=build(study)
    report=check(root,study,plan,require_complete=False)
    assert report['request_dependency_bindings_checked']
    assert not report['scientific_verification'] and not report['complete']
    with pytest.raises(ValueError,match='complete'):check(root,study,plan)
    path=root/'attempt_000001.json'
    original=json.loads(path.read_text())
    for change in ('unit','callback','dependency'):
        artifact=json.loads(json.dumps(original))
        request=artifact['worker_receipt']['request']
        if change=='unit':request['arguments']['unit']['days']=0
        elif change=='callback':request['function']='another_callback'
        else:request['arguments']['dependencies']=[dict(item='not_committed')]
        artifact['worker_receipt']['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
        path.write_text(json.dumps(artifact))
        with sqlite3.connect(root/'attempts.sqlite') as db:
            db.execute('UPDATE attempts SET sha256=? WHERE id=1',(digest(path),))
        with pytest.raises(ValueError):check(root,study,plan,require_complete=False)
