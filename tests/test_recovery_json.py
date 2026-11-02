import json
import pytest
from src.experiments import recovery_json as fast
from src.experiments.execution_journal import atomic_json as original


def test_identical_bytes_and_fsync_before_replace(tmp_path,monkeypatch):
    value={'unicode':'₹','values':[None,True,1.25,1e-12,{'x':'line\nbreak'}]}
    original(tmp_path/'reference',value)
    calls=[];fsync=fast.os.fsync;replace=fast.os.replace
    def sync(fd):calls.append('fsync');fsync(fd)
    def move(a,b):calls.append('replace');replace(a,b)
    monkeypatch.setattr(fast.os,'fsync',sync);monkeypatch.setattr(fast.os,'replace',move)
    fast.atomic_json(tmp_path/'actual',value)
    assert (tmp_path/'actual').read_bytes()==(tmp_path/'reference').read_bytes()
    assert calls==['fsync','replace']


def test_invalid_payload_and_existing_temporary_fail_closed(tmp_path):
    path=tmp_path/'result.json';path.write_text('original')
    with pytest.raises(ValueError):fast.atomic_json(path,{'bad':float('nan')})
    assert path.read_text()=='original'
    path.with_name(path.name+'.tmp').write_text('unresolved')
    with pytest.raises(FileExistsError):fast.atomic_json(path,{'good':1})
    assert path.read_text()=='original'
