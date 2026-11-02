"""Inventory a terminated rollout screen without fitting or new simulation."""
import argparse
import json
from pathlib import Path
from scripts.train_compact_residual import digest,write_json


def run(source,output):
    root=Path(source);out=Path(output)
    manifest=json.loads((root/'manifest.json').read_text())
    failure=json.loads((root/'failed_status.json').read_text())
    assert failure['stage']=='fit' and failure['error']=='conditional pricing time cap'
    for p,h in manifest['hashes'].items():assert digest(p)==h,p
    assert not any((root/p).exists() for p in ['policy.json','assessment_freeze.json','assessment.jsonl','summary.json'])
    state_ids=[];original_ids=[];saved_days=0;suffixes=0
    with (root/'fit_states.jsonl').open() as handle:
        for line in handle:
            r=json.loads(line);state_ids.append([r['episode'],r['day']])
            assert r['before']==r['after'] and len(r['draws'])==16 and len(r['replays'])==2
            length=29-r['day'];assert len(r['probe'])==length
            saved_days+=length
            for draw in r['draws']:
                assert len(draw['branches'])==11
                for b in draw['branches']:
                    assert len(b['trace'])==length
                    assert b['value']==sum(.99**i*v['reward'] for i,v in enumerate(b['trace']))
                    saved_days+=length;suffixes+=1
            for replay in r['replays']:
                assert len(replay['trace'])==length;saved_days+=length
    with (root/'originals.jsonl').open() as handle:
        for line in handle:
            r=json.loads(line);original_ids.append(r['episode']);assert len(r['trace'])==28;saved_days+=28
    expected=[[e,d] for e in range(16) for d in [1,8,15,22]]
    assert state_ids==expected[:len(state_ids)] and original_ids==list(range(len(original_ids)))
    result=dict(status='audited_execution_failure',failure=failure,complete_states=len(state_ids),
        complete_originals=len(original_ids),complete_archived_suffixes=suffixes,
        archived_simulator_days=saved_days,executed_days=failure['counters']['days'],
        executed_days_without_complete_archived_records=failure['counters']['days']-saved_days,
        next_unsaved_state=expected[len(state_ids)] if len(state_ids)<64 else None,
        source_hashes_unchanged=True,complete_records_form_declared_prefix=True,
        new_simulator_days=0,new_policy_fits=0,
        limitation='Structural prefix inventory and exact archived values only; not full action/economic verification or a scientific outcome. No permission to resume or extend the failed protocol.',
        hashes={p:digest(root/p) for p in ['manifest.json','failed_status.json','fit_states.jsonl','originals.jsonl']})
    assert result['executed_days_without_complete_archived_records']>=0
    out.mkdir(parents=True,exist_ok=False);write_json(out/'audit.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.source,a.output)
