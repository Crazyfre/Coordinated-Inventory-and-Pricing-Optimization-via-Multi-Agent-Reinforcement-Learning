def test_runner_rejects_unbounded_updates_before_writing(tmp_path):
    from scripts.diagnose_ordering_learning import run
    import pytest
    with pytest.raises(ValueError, match='bounded'):
        run(str(tmp_path / 'bad'), updates=129)
    assert not (tmp_path / 'bad').exists()
