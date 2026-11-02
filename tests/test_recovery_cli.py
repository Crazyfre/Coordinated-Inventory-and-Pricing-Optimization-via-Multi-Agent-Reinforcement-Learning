import hashlib
import json
from pathlib import Path
import pytest
from scripts import run_recovery_campaign as cli
from scripts.recovery_campaign_plan import build
from tests.test_recovery_campaign_plan import study


def packet(tmp_path):
    config = study()
    plan = build(config)
    value = dict(root=str(Path.cwd()/'outputs'/'cli_synthetic_unused'),
        declaration=dict(study=config, protocol='synthetic', runtime={},
            schedule=[[0, len(plan['units'])]], overhead={}), plan=plan)
    path = tmp_path/'packet.json'
    return value, path


def save(value, path):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_external_checksum_and_plan_tampering(tmp_path):
    value, path = packet(tmp_path)
    checksum = save(value, path)
    assert cli.read_packet(path, checksum) == value
    with pytest.raises(ValueError, match='checksum'):
        cli.read_packet(path, '0'*64)
    value['plan']['units'][0]['days'] += 1
    checksum = save(value, path)
    with pytest.raises((ValueError, AssertionError)):
        cli.read_packet(path, checksum)


def test_cli_passes_original_entry_time_and_exact_declaration(tmp_path, monkeypatch):
    value, path = packet(tmp_path)
    checksum = save(value, path)
    from scripts import recovery_invocation as invocation
    calls = []
    def launch(root, declaration, started_at):
        calls.append((root, declaration, started_at))
        return {'status': 'synthetic'}
    monkeypatch.setattr(invocation, 'launch', launch)
    cli.main(['run', '--packet', str(path), '--sha256', checksum])
    assert calls == [(value['root'], value['declaration'], cli.ENTRY_STARTED)]
    value['root'] = 'relative/archive'
    checksum = save(value, path)
    with pytest.raises(ValueError, match='absolute'):
        cli.read_packet(path, checksum)
