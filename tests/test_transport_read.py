from pathlib import Path
import pytest
from src.experiments.transport_read import read_bytes


def test_access_retry_reads_once_successful_without_extending_deadline(monkeypatch):
    now=[0.];calls=[]
    def read(path):
        calls.append(path)
        if len(calls)<3:raise PermissionError('transient')
        return b'unchanged payload'
    monkeypatch.setattr(Path,'read_bytes',read)
    assert read_bytes('fixture',.1,clock=lambda:now[0],sleep=lambda s:now.__setitem__(0,now[0]+s))==b'unchanged payload'
    assert len(calls)==3 and now[0]==.02


def test_persistent_denial_stops_at_original_deadline(monkeypatch):
    now=[0.]
    def read(path):raise PermissionError('persistent')
    monkeypatch.setattr(Path,'read_bytes',read)
    with pytest.raises(TimeoutError):
        read_bytes('fixture',.025,clock=lambda:now[0],sleep=lambda s:now.__setitem__(0,now[0]+s))
    assert now[0]==.025


def test_other_errors_are_not_retried(monkeypatch):
    def read(path):raise FileNotFoundError('missing')
    monkeypatch.setattr(Path,'read_bytes',read)
    with pytest.raises(FileNotFoundError):read_bytes('fixture',1,clock=lambda:0,sleep=lambda s:pytest.fail('retry'))
