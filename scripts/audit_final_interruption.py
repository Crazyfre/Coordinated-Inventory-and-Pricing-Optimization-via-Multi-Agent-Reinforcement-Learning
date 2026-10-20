"""Read-only count/hash audit of an externally interrupted held-out run."""
import json
from pathlib import Path
from scripts.train_compact_residual import digest,write_json


def run():
    root=Path('outputs/structured_final/evaluation_v1')
    manifest=json.loads((root/'frozen_manifest.json').read_text())
    for path,h in manifest['hashes'].items():assert digest(path)==h
    for policy in manifest['policies'].values():assert digest(policy['path'])==policy['sha256']
    labels=list(manifest['policies'])+manifest['baseline_labels']
    required=[(label,key) for key in manifest['keys'] for label in labels]
    counts=[];total=0
    for path in sorted(root.glob('*_traces.jsonl')):
        complete=[];current=[];key=None;malformed=0;last_offset=0;complete_bytes=0
        with path.open('rb') as handle:
            for raw in handle:
                last_offset+=len(raw)
                try:row=json.loads(raw)
                except (ValueError,UnicodeDecodeError):malformed+=1;continue
                assert malformed==0,'data after malformed record'
                new=(row['label'],row['scenario'])
                if key is None:key=new
                assert new==key,'incomplete episode before another key'
                current.append(row)
                assert row['day']==len(current) and row['start']==1914
                if len(current)==28:
                    assert key==required[len(complete)]
                    complete.append(key);current=[];key=None;complete_bytes=last_offset
        counts.append(dict(file=path.name,sha256=digest(path),complete_episodes=len(complete),
            complete_bytes=complete_bytes,partial_rows=len(current),partial_key=key,
            malformed_lines=malformed,last_complete=complete[-1] if complete else None))
        total+=len(complete)
    record=dict(status='audited',frozen_manifest_sha256=digest(root/'frozen_manifest.json'),
        sources_unchanged=True,checkpoints_unchanged=True,files=counts,
        complete_episodes=total,complete_days=total*28,new_simulator_days=0)
    write_json(Path('outputs/structured_final/evaluation_interruption_audit.json'),record)
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':run()
