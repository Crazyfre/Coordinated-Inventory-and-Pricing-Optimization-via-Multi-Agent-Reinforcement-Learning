import json
import pytest
from src.experiments.execution_journal import ExecutionJournal


def test_committed_progress_reopens_but_pending_work_cannot_retry(tmp_path):
    root=tmp_path/'run';j=ExecutionJournal(root,'protocol-source-hash',['a','b'],10)
    j.claim('a',4);artifact=root/'a.json';artifact.write_text('{}');j.commit(artifact,4)
    reopened=ExecutionJournal(root,'protocol-source-hash',['a','b'],10)
    assert reopened.state['reserved_days']==4
    with pytest.raises(ValueError):reopened.claim('a',4)
    with pytest.raises(ValueError):reopened.claim('b',7)
    reopened.claim('b',6)
    # Simulated process loss after durable reservation, before commit.
    with pytest.raises(RuntimeError,match='interrupted'):ExecutionJournal(root,'protocol-source-hash',['a','b'],10)
    saved=json.loads((root/'journal.json').read_text());assert saved['reserved_days']==10 and saved['pending']['item']=='b'


def test_plan_artifact_and_incomplete_commit_rejected(tmp_path):
    root=tmp_path/'run';j=ExecutionJournal(root,'frozen',['a'],5);j.claim('a',5)
    artifact=root/'a.json';artifact.write_text('trace')
    with pytest.raises(ValueError):j.commit(artifact,4)
    j.commit(artifact,5)
    with pytest.raises(ValueError):ExecutionJournal(root,'changed',['a'],5)
    artifact.write_text('tampered')
    with pytest.raises(ValueError,match='artifact'):ExecutionJournal(root,'frozen',['a'],5)
