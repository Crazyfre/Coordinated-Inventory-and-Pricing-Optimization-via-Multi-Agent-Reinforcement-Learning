import json
import subprocess
import pytest
from scripts.finish_structured_final import execute_stages,run


def test_failed_verification_prevents_evaluation(tmp_path):
    calls=[]
    def fail(args,**kwargs):
        calls.append(args)
        raise subprocess.CalledProcessError(1,args)
    with pytest.raises(subprocess.CalledProcessError):
        execute_stages([('verify',['verify']),('evaluate',['evaluate'])],tmp_path,run=fail)
    assert len(calls)==1 and not (tmp_path/'evaluate.log').exists()
    assert json.loads((tmp_path/'progress.json').read_text())['completed']==[]


def test_existing_evaluation_rejected_before_process_attach(tmp_path):
    target=tmp_path/'evaluation';target.mkdir()
    with pytest.raises(FileExistsError):run(0,tmp_path/'handoff',target)
    assert not (tmp_path/'handoff').exists()
