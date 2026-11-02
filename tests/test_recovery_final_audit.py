import pytest
from scripts import verify_recovery_campaign as audit


def setup(monkeypatch):
    calls=[]
    declaration=dict(study={},protocol='frozen',runtime={'version':'fixed'},schedule=[],overhead={})
    monkeypatch.setattr(audit,'check_plan',lambda *a:calls.append('plan'))
    monkeypatch.setattr(audit,'validate_coverage',lambda *a:calls.append('coverage'))
    monkeypatch.setattr(audit,'runtime_identity',lambda:declaration['runtime'])
    execution=dict(invocation_accounting_checked=True,overhead_limits_checked=True,combined_budget_checked=True)
    def invocations(*args):calls.append('execution');return execution
    monkeypatch.setattr(audit,'check_invocations',invocations)
    science=dict(status='passed',scientific_verification=True,primary_pass=False,
                 representation_pass=False,heuristic_pass=False)
    def scientific(*args):calls.append('science');return science
    monkeypatch.setattr(audit,'verify_archive',scientific)
    return declaration,calls,execution,science


def test_combined_audit_preserves_negative_science_and_full_goal(monkeypatch,tmp_path):
    declaration,calls,_,_=setup(monkeypatch)
    result=audit.verify(tmp_path,declaration,{})
    assert calls==['plan','coverage','execution','science']
    assert result['archive_verified'] and not result['pricing_gate_passed']
    assert not result['full_project_goal_achieved'] and not result['joint_mappo_verified']
    assert not list(tmp_path.iterdir())  # read-only; no false verification marker


@pytest.mark.parametrize('key',['invocation_accounting_checked','overhead_limits_checked','combined_budget_checked'])
def test_missing_budget_checks_stop_before_science(monkeypatch,tmp_path,key):
    declaration,calls,execution,_=setup(monkeypatch);execution[key]=False
    with pytest.raises(ValueError,match='budget'):audit.verify(tmp_path,declaration,{})
    assert 'science' not in calls


def test_runtime_mismatch_and_scientific_failure_propagate(monkeypatch,tmp_path):
    declaration,calls,_,science=setup(monkeypatch)
    monkeypatch.setattr(audit,'runtime_identity',lambda:{'version':'different'})
    with pytest.raises(ValueError,match='runtime'):audit.verify(tmp_path,declaration,{})
    assert 'execution' not in calls
    monkeypatch.setattr(audit,'runtime_identity',lambda:declaration['runtime'])
    science['scientific_verification']=False
    with pytest.raises(ValueError,match='scientific'):audit.verify(tmp_path,declaration,{})
