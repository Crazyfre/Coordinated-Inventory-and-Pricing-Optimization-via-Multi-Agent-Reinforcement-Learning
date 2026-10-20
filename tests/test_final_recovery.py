import pytest
from scripts.recover_structured_evaluation import validate_prefix


def test_only_complete_audited_prefix_can_continue():
    record=dict(file='S1_traces.jsonl',partial_rows=0,malformed_lines=0,complete_episodes=703)
    assert validate_prefix(dict(files=[record]))==record
    for changed in [dict(partial_rows=1),dict(malformed_lines=1),dict(complete_episodes=1700),dict(file='S2_traces.jsonl')]:
        with pytest.raises(ValueError):validate_prefix(dict(files=[dict(record,**changed)]))
