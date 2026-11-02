"""Dispatcher for prospective fresh-key work units. Verifier required before launch."""
import time
ENTRY_STARTED=time.perf_counter()
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scripts import cross_sku_segmented_units as units
from scripts.train_compact_residual import INPUTS,serial
from scripts.diagnose_conditional_pricing import STUDY
from src.experiments.execution_journal import atomic_json,digest
from src.experiments.segmented_execution import execute_one
from src.experiments.research import make_research_env
from src.demand.identifiable_calibration import load_identifiable_training_calibration

CAPS=dict(fit=7200.,assessment=600.,total=7800.)
MAXIMUM_DAYS=501872


def identity(manifest):
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,allow_nan=False).encode()).hexdigest()


def new_manifest():
    paths={str(p) for folder in ['src','scripts'] for p in Path(folder).rglob('*.py')}
    paths|={str(p) for p in Path('data').rglob('*') if p.is_file() and ('aligned' in str(p) or 'sku' in p.name.lower())}
    paths|={'src/config/skus.json',INPUTS['calibration'],str(STUDY/'sku_seed211_final.json'),
        'CROSS_SKU_SEGMENTED_PROTOCOL.md','CROSS_SKU_PRICING_PROTOCOL.md','CROSS_SKU_FEATURE_SPEC.md',
        'outputs/structured_mappo/cross_sku_interruption_v1/audit.json'}
    audit=json.loads(Path('outputs/structured_mappo/cross_sku_interruption_v1/audit.json').read_text())
    assert audit['status']=='interrupted_archive_inventoried' and audit['source_hashes_unchanged']
    return dict(hashes={p:digest(p) for p in sorted(paths)},items=units.plan(),caps=CAPS,maximum_days=MAXIMUM_DAYS,
        covers=json.loads((STUDY/'sku_seed211_final.json').read_text())['mean'])


def read_committed(root,item):
    root=Path(root);journal=json.loads((root/'journal.json').read_text())
    matches=[r for r in journal['completed'] if r['item']==item]
    if len(matches)!=1:raise ValueError('required committed dependency missing')
    row=matches[0];path=root/row['artifact']
    if not path.resolve().is_relative_to(root.resolve()) or digest(path)!=row['sha256']:
        raise ValueError('dependency path/hash mismatch')
    value=json.loads(path.read_text())
    if value['item']!=item or value['days']!=row['days']:raise ValueError('dependency metadata mismatch')
    return value['payload'],row['sha256']


def reservation(root,item):
    if item.startswith('scan_'):return 'fit',28
    if item.startswith('branch_'):
        selected,_=read_committed(root,'selection');ep=int(item.split('_')[1])
        return 'fit',28+67*sum(29-d for d,i in selected['selections'][ep]['selected'])
    if item in ['selection','fit']:return 'fit',0
    if item.startswith('assess_') or item=='replays':return 'assessment',112
    if item=='summary':return 'assessment',0
    raise ValueError('unknown work unit')


def perform(root,item,manifest,deadline):
    load=lambda name:read_committed(root,name)[0]
    if item=='selection':
        scans=[];hashes=[]
        for e in range(128):
            s,h=read_committed(root,f'scan_{e:03d}');scans.append(s);hashes.append(h)
        base=scans[0]['base_prices']
        if any(s['base_prices']!=base for s in scans):raise ValueError('base prices changed across scans')
        return units.freeze(scans,base,hashes,manifest['hashes'])
    if item=='fit':
        branches=[];hashes=[]
        for e in range(128):
            b,h=read_committed(root,f'branch_{e:03d}')
            # Keep samples only: loading all suffix traces together is unnecessary.
            branches.append(dict(episode=b['episode'],states=[dict(sample=s['sample']) for s in b['states']]))
            hashes.append(h)
        _,selection_hash=read_committed(root,'selection')
        return units.fit_models(branches,load('scan_000')['base_prices'],selection_hash,hashes,manifest['hashes'])
    if item=='summary':
        selected=load('selection');journal=json.loads((Path(root)/'journal.json').read_text())
        days=sum(r['days'] for r in journal['completed'])
        if days!=selected['planned_days']:raise ValueError('full campaign day count mismatch')
        assessment=[]
        for e in range(128):
            a=load(f'assess_{e:03d}')
            assessment.append(dict(episode=e,records=[dict(result=r['result']) for r in a['records']]))
        return dict(status='complete',analysis=units.analyze(assessment),days=days,
                    suffix_length_sum=selected['suffix_length_sum'],test_scoring=False),0
    covers=np.array(manifest['covers']);torch.set_num_threads(1)
    env=make_research_env('S2',split='train',start=15,pricing_scope='per_sku',
        params=load_identifiable_training_calibration(INPUTS['calibration']))
    try:
        if item.startswith('scan_'):
            result,days=units.scan(env,covers,int(item.split('_')[1]),deadline)
            result['base_prices']=env.base_price.tolist();return result,days
        if item.startswith('branch_'):
            ep=int(item.split('_')[1]);selected=load('selection')
            return units.branch(env,covers,load(f'scan_{ep:03d}'),selected['selections'][ep],deadline)
        models=load('fit')['models']
        if item=='replays':return units.replays(env,covers,models,load('assess_000'),deadline)
        if item.startswith('assess_'):return units.assessment(env,covers,int(item.split('_')[1]),models,deadline)
        raise ValueError('unknown work unit')
    finally:env.close()


def run_one(output,started_at=None):
    began=time.perf_counter() if started_at is None else started_at;root=Path(output)
    if root.exists():
        manifest=json.loads((root/'manifest.json').read_text())
        journal=json.loads((root/'journal.json').read_text());index=len(journal['completed'])
    else:manifest=new_manifest();index=0
    if manifest['items']!=units.plan() or manifest['caps']!=CAPS or manifest['maximum_days']!=MAXIMUM_DAYS:
        raise ValueError('declared campaign plan mismatch')
    if index>=len(manifest['items']):raise ValueError('all work already committed')
    item=manifest['items'][index];phase,days=reservation(root,item)
    def work(path,deadline):
        for source,h in manifest['hashes'].items():
            if digest(source)!=h:raise ValueError(f'frozen source changed: {source}')
        if index==0:
            if (root/'manifest.json').exists():raise RuntimeError('uncommitted manifest exists')
            atomic_json(root/'manifest.json',manifest)
        elif identity(json.loads((root/'manifest.json').read_text()))!=identity(manifest):
            raise ValueError('manifest changed')
        result,actual=perform(root,item,manifest,deadline)
        atomic_json(path,serial(dict(item=item,days=actual,payload=result)))
        return actual
    return execute_one(root,identity(manifest),manifest['items'],MAXIMUM_DAYS,CAPS,item,phase,days,work,started_at=began)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--units',type=int,default=1)
    args=p.parse_args()
    if not 1<=args.units<=388:p.error('--units must be1..388')
    for i in range(args.units):
        print(json.dumps(run_one(args.output,ENTRY_STARTED if i==0 else None)),flush=True)
