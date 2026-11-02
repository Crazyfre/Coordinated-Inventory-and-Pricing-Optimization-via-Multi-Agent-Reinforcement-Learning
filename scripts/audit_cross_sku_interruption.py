"""Inventory an interrupted archive without resuming collection or fitting."""
import argparse
import json
from pathlib import Path
from scripts.train_compact_residual import digest,write_json


def run(source,output):
    root=Path(source);m=json.loads((root/'manifest.json').read_text())
    progress=json.loads((root/'progress.json').read_text());sel=json.loads((root/'selection_freeze.json').read_text())
    assert not any((root/n).exists() for n in ['policy.json','assessment_freeze.json','summary.json','assessment.jsonl'])
    changed=[p for p,h in m['hashes'].items() if not Path(p).exists() or digest(p)!=h]
    assert sel['source_hashes']==m['hashes'] and digest(root/'scan.jsonl')==sel['scan_sha256']
    scans={}
    with (root/'scan.jsonl').open() as f:
        for line in f:
            r=json.loads(line);assert r['episode'] not in scans;scans[r['episode']]=r
    assert sorted(scans)==list(range(128))
    days=sum(len(r['trace']) for r in scans.values());assert days==3584
    expected=[]
    for ep,s in enumerate(sel['selections']):
        assert s['episode']==ep
        expected.extend((ep,rank,day,sku) for rank,(day,sku) in sorted(enumerate(s['selected']),key=lambda z:(z[1][0],z[0])))
    ids=[];suffixes=0;truncated=None
    with (root/'states.jsonl').open() as f:
        for line_index,line in enumerate(f):
            try:r=json.loads(line)
            except json.JSONDecodeError:
                assert not f.read();truncated=line_index;break
            s=r['sample'];key=(s['episode'],s['rank'],s['day'],s['sku']);ids.append(key)
            assert key==expected[len(ids)-1] and r['before']==r['after']
            length=29-s['day'];assert r['probe']==scans[s['episode']]['trace'][s['day']-1:]
            assert len(r['draws'])==32 and len(r['replays'])==2;gains=[];days+=length
            for rep,draw in enumerate(r['draws']):
                assert [b['action'] for b in draw['branches']]==[0,4];values=[]
                for j,b in enumerate(draw['branches']):
                    assert len(b['trace'])==length and b['initial']==r['before']
                    value=sum(.99**i*t['reward'] for i,t in enumerate(b['trace']));assert value==b['value']
                    values.append(value);days+=length;suffixes+=1
                    if rep==0:assert r['replays'][j]==b['trace']
                gains.append(values[0]-values[1])
            assert gains==s['gains'];days+=2*length
    originals=[]
    with (root/'original_replays.jsonl').open() as f:
        for line in f:
            r=json.loads(line);assert r==scans[len(originals)];originals.append(r['episode']);days+=28
    assert len(originals)>=progress['windows']
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    result=dict(status='interrupted_archive_inventoried',complete_states=len(ids),complete_original_replays=len(originals),
        complete_suffixes=suffixes,complete_archived_days=days,last_progress=progress,
        changed_or_missing_sources=changed,source_hashes_unchanged=not changed,
        truncated_final_state_line=truncated,next_unsaved_state=expected[len(ids)] if len(ids)<len(expected) else None,
        total_executed_days_known=False,termination_cause='unknown; no failure marker or exit status available',
        new_simulator_days=0,policy_fits=0,
        limitation='Structural prefix and archived return audit only, not full scientific verification. Saved days exclude unrecorded execution; no resume/refit authorization.',
        hashes={n:digest(root/n) for n in ['manifest.json','selection_freeze.json','progress.json','scan.jsonl','states.jsonl','original_replays.jsonl']})
    write_json(out/'audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.source,a.output)
