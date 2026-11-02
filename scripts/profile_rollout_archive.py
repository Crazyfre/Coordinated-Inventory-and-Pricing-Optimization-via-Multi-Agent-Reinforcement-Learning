"""Bounded archive I/O benchmark; no environments, fits, or outcome selection."""
import argparse
import json
import time
from pathlib import Path
from scripts.train_compact_residual import digest,write_json


def run(source,output):
    root=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    audit=json.loads(Path('outputs/structured_mappo/rollout_pricing_timeout_audit_v1/audit.json').read_text())
    path=root/'fit_states.jsonl';started=time.perf_counter();h=digest(path);hash_seconds=time.perf_counter()-started
    assert h==audit['hashes']['fit_states.jsonl']
    with path.open('rb') as f:
        started=time.perf_counter();line=f.readline();read_seconds=time.perf_counter()-started
    times=[]
    for _ in range(3):
        t=time.perf_counter();record=json.loads(line);parse=time.perf_counter()-t
        t=time.perf_counter();encoded=json.dumps(record,allow_nan=False);encode=time.perf_counter()-t
        assert json.loads(encoded)==record
        times.append(dict(parse_seconds=parse,encode_seconds=encode))
    assert record['episode']==0 and record['day']==1
    result=dict(status='complete',archive_bytes=path.stat().st_size,first_record_bytes=len(line),
        hash_seconds=hash_seconds,first_record_read_seconds=read_seconds,three_repetitions=times,
        new_simulator_days=0,new_policy_fits=0,archive_sha256=h,
        limitation='First fixed record only; warm-cache microbenchmark, not end-to-end attribution. Simulator/deepcopy cost is not measured. No outcomes used for selection.')
    write_json(out/'profile.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.source,a.output)
