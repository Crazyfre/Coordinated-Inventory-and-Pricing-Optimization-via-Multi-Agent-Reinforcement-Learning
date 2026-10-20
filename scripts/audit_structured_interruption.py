"""Read-only accounting of the interrupted first structured training run."""
import json
from pathlib import Path
from scripts.train_compact_residual import digest,write_json


def audit():
    root=Path('outputs/structured_final/training_v1');folder=root/'main_266'
    history=json.loads((folder/'history.json').read_text());valid=0;bad=[];last=None
    with (folder/'traces.jsonl').open() as handle:
        for i,line in enumerate(handle):
            try:
                r=json.loads(line);valid+=1;last={k:r[k] for k in ['label','scenario','day']}
            except json.JSONDecodeError:bad.append(dict(line=i+1,bytes=len(line)))
    result=dict(saved_updates=len(history),valid_daily_rows=valid,last=last,malformed_lines=bad,
        saved_update_days=len(history)*8*2*28,source_hashes={str(p):digest(p) for p in folder.iterdir() if p.is_file()},
        original_manifest_sha256=digest(root/'manifest.json'),new_simulator_days=0)
    target=Path('outputs/structured_final/interruption_audit.json')
    if target.exists():raise FileExistsError(target)
    write_json(target,result);print(json.dumps(result,indent=2))


if __name__=='__main__':audit()
