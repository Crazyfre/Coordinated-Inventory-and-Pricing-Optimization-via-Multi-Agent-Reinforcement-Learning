"""Verify exact interruption-prefix reuse plus the complete original cohort."""
import json
from pathlib import Path
from scripts.train_compact_residual import digest,write_json
from scripts.verify_structured_training import verify


def run():
    root=Path('outputs/structured_final/training_recovered_v1');source=Path('outputs/structured_final/training_v1')
    manifest=json.loads((root/'manifest.json').read_text());audit=json.loads(Path('outputs/structured_final/interruption_audit.json').read_text())
    assert digest(source/'manifest.json')==manifest['recovery']['original_manifest_sha256']
    for p,h in audit['source_hashes'].items():assert digest(p)==h
    with (source/'main_266/traces.jsonl').open('rb') as old,(root/'main_266/traces.jsonl').open('rb') as new:
        while True:
            chunk=old.read(1024*1024)
            if not chunk:break
            assert chunk==new.read(len(chunk))
    before=json.loads((source/'main_266/history.json').read_text())
    after=json.loads((root/'main_266/history.json').read_text())
    assert before==after[:36]
    verify(root)
    summary=json.loads((root/'summary.json').read_text())
    assert summary['newly_simulated']==dict(episodes=25015,days=700420)
    assert summary['retained']==dict(episodes=585,days=16380)
    result=json.loads((root/'verification.json').read_text())
    result.update(exact_preserved_prefix=True,complete_episodes_repeated=0,
        recovery_accounting_verified=True,original_seed266_total_timing='unavailable; not claimed compliant',
        recovery_verifier_sha256=digest(__file__))
    write_json(root/'verification.json',result)
    print('Exact prefix recovery and full cohort verified.',flush=True)


if __name__=='__main__':run()
