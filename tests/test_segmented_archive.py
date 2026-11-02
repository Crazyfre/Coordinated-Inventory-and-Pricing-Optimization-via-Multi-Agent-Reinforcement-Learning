import hashlib
import json
import pytest
from scripts.check_segmented_archive import check_archive, sha256


def put(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def archive(tmp_path):
    root = tmp_path / 'archive'; root.mkdir()
    source = tmp_path / 'source.py'; source.write_text('fixture')
    items = ([f'scan_{e:03d}' for e in range(128)] + ['selection'] +
             [f'branch_{e:03d}' for e in range(128)] + ['fit'] +
             [f'assess_{e:03d}' for e in range(128)] + ['replays', 'summary'])
    caps = dict(fit=7200., assessment=600., total=7800.)
    hashes = {str(source): sha256(source)}
    manifest = dict(items=items, caps=caps, maximum_days=501872, hashes=hashes)
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()
    selections = [dict(episode=e, key=2570000+e, selected=[[1, 0], [22, 9]]) for e in range(128)]
    length = 128 * 35; total = 21616 + 67 * length
    entries = []
    for index, item in enumerate(items):
        days = 0
        if item.startswith('scan_'):
            e = int(item[5:]); payload = dict(episode=e, key=2570000+e); days=28
        elif item == 'selection':
            payload = dict(source_hashes=hashes, scan_hashes=[r['sha256'] for r in entries],
                           selections=selections, suffix_length_sum=length, planned_days=total)
        elif item.startswith('branch_'):
            e = int(item[7:]); days = 28 + 67 * 35
            payload = dict(episode=e, key=2570000+e, states=[dict(sample=dict(episode=e, rank=r, day=d, sku=s))
                for r, (d,s) in enumerate([[1,0],[22,9]])])
        elif item == 'fit':
            payload = dict(source_hashes=hashes, selection_hash=entries[128]['sha256'],
                branch_hashes=[r['sha256'] for r in entries[129:257]], models=dict(local={},cross={}),
                samples=[dict(episode=e,rank=r) for e in range(128) for r in [0,1]])
        elif item.startswith('assess_') or item == 'replays':
            e = 0 if item == 'replays' else int(item[7:]); days=112
            payload = dict(episode=e,records=[dict(label=l,key=2590000+e) for l in ['cross','local','baseline','heuristic']])
        else:
            payload = dict(status='complete',test_scoring=False,days=total,suffix_length_sum=length)
        name = f'unit_{index:04d}.json'; put(root/name,dict(item=item,days=days,payload=payload))
        entries.append(dict(item=item,days=days,artifact=name,sha256=sha256(root/name)))
    put(root/'manifest.json',manifest)
    put(root/'journal.json',dict(plan=dict(identity=identity,items=items,maximum_days=501872),
        pending=None,completed=entries,reserved_days=total))
    history=[dict(item=item,phase='fit' if i<=257 else 'assessment',elapsed=12/258 if i<=257 else 2/130,
        start_wall=float(i*10),end_wall=float(i*10+1),boundary_gap_seconds=None if i==0 else 9.) for i,item in enumerate(items)]
    put(root/'time.json',dict(plan=dict(identity=identity,caps=caps),pending=None,status='idle',
        completed=items,used=dict(fit=12.,assessment=2.,total=14.),history=history))
    return root


def test_complete_structure_does_not_claim_scientific_verification(tmp_path):
    result = check_archive(archive(tmp_path))
    assert result['units'] == 388 and result['days'] == 321776
    assert result['scientific_verification'] is False and result['new_simulator_days'] == 0


@pytest.mark.parametrize('change', ['pending','missing','order','budget','sum','hash','freeze','days','namespace','model','history','gap'])
def test_archive_rejects_corruption(tmp_path, change):
    root = archive(tmp_path)
    journal=json.loads((root/'journal.json').read_text()); timer=json.loads((root/'time.json').read_text())
    if change == 'pending': timer['pending'] = dict(item='summary')
    elif change == 'missing': journal['completed'].pop()
    elif change == 'order': journal['completed'][0],journal['completed'][1]=journal['completed'][1],journal['completed'][0]
    elif change == 'budget': timer['used']['assessment']=601.
    elif change == 'sum': timer['used']['total']=15.
    elif change == 'hash': (root/'unit_0000.json').write_text('{}')
    elif change == 'history': timer['history'][0]['elapsed']+=1
    elif change == 'gap': timer['history'][1]['boundary_gap_seconds']=0
    else:
        index = dict(freeze=257,days=387,namespace=258,model=257)[change]
        path=root/f'unit_{index:04d}.json'; value=json.loads(path.read_text())
        if change == 'freeze': value['payload']['branch_hashes'][0]='wrong'
        elif change == 'days': value['days']=1; journal['completed'][index]['days']=1
        elif change == 'namespace': value['payload']['records'][0]['key']=2560000
        elif change == 'model': del value['payload']['models']['local']
        put(path,value); journal['completed'][index]['sha256']=sha256(path)
    put(root/'journal.json',journal);put(root/'time.json',timer)
    with pytest.raises(ValueError):check_archive(root)
