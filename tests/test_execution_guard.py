import json
import pytest
from src.experiments.execution_guard import RunLock,CumulativeClock


def test_exclusive_lock_released_on_exception(tmp_path):
    path=tmp_path/'owner.lock'
    with pytest.raises(ValueError):
        with RunLock(path):
            with pytest.raises(RuntimeError):
                with RunLock(path):pass
            raise ValueError('simulated failure')
    with RunLock(path):pass


def test_cumulative_budget_survives_clean_boundary_and_records_failure(tmp_path):
    now=[0.];clock=lambda:now[0];path=tmp_path/'time.json';caps=dict(fit=10.,assessment=4.,total=12.)
    c=CumulativeClock(path,'same',caps,clock);assert c.begin('scan','fit')==10
    now[0]=7;c.finish()
    now[0]=100 # idle gap does not reset or spend active budget
    c=CumulativeClock(path,'same',caps,clock);assert c.begin('branch','fit')==103
    now[0]=104
    with pytest.raises(TimeoutError):c.finish()
    saved=json.loads(path.read_text());assert saved['used']['fit']==11 and saved['status']=='failed'
    with pytest.raises(RuntimeError):CumulativeClock(path,'same',caps,clock)


def test_unfinished_unit_and_changed_plan_rejected(tmp_path):
    path=tmp_path/'time.json';caps=dict(fit=10.,assessment=4.,total=12.)
    c=CumulativeClock(path,'same',caps);c.begin('fit','fit')
    with pytest.raises(RuntimeError):CumulativeClock(path,'same',caps)
    with pytest.raises(ValueError):CumulativeClock(path,'different',caps)
    c.finish(error='worker terminated')
    assert json.loads(path.read_text())['error']=='worker terminated'


def test_finalization_write_is_charged_and_boundary_gaps_disclosed(tmp_path,monkeypatch):
    from src.experiments import execution_guard as guard
    now=[0.];wall=[1000.];original=guard.atomic_json
    def write(path,state):
        original(path,state)
        if state['status']=='finalizing':now[0]+=2.;wall[0]+=2.
    monkeypatch.setattr(guard,'atomic_json',write)
    path=tmp_path/'time.json';caps=dict(fit=20.,assessment=10.,total=30.)
    c=CumulativeClock(path,'id',caps,lambda:now[0],lambda:wall[0])
    c.begin('a','fit');now[0]+=3;wall[0]+=3;c.finish()
    assert c.state['used']['fit']==5.
    now[0]+=100;wall[0]+=100
    c=CumulativeClock(path,'id',caps,lambda:now[0],lambda:wall[0])
    c.begin('b','assessment');now[0]+=1;wall[0]+=1;c.finish()
    assert c.state['used']==dict(fit=5.,assessment=3.,total=8.)
    assert c.state['history'][1]['boundary_gap_seconds']==100.


def test_interrupted_finalization_refuses_continuation(tmp_path,monkeypatch):
    from src.experiments import execution_guard as guard
    original=guard.atomic_json
    def crash(path,state):
        original(path,state)
        if state['status']=='finalizing':raise OSError('simulated receipt interruption')
    path=tmp_path/'time.json';caps=dict(fit=10.,assessment=4.,total=12.)
    c=CumulativeClock(path,'id',caps);c.begin('a','fit')
    monkeypatch.setattr(guard,'atomic_json',crash)
    with pytest.raises(OSError):c.finish()
    assert json.loads(path.read_text())['status']=='finalizing'
    with pytest.raises(RuntimeError):CumulativeClock(path,'id',caps)
