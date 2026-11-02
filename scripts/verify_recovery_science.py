"""Independent recovery scientific checks; no simulator or producer imports.

Full read-only archive orchestration returns evidence only after every check.
No persistent verification marker is written by this module.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
from src.config.sku_config import load_skus
from scripts.check_recovery_campaign_archive import check as check_archive
from scripts.verify_cross_sku_segmented import feature_matrix,check_ridge,audit_trace,original_noise_keys


def hashed(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def check_plan(study,plan):
    assert plan['identity']==hashed(study) and plan['caps']==study['caps'] and plan['max_attempts']==3
    specs=[]
    def entry(name,kind,days,deps,**fields):
        specs.append((dict(item=name,kind=kind,dependencies=deps,**fields),days))
    scans=[f'scan_{e:03d}' for e in range(128)]
    for e,name in enumerate(scans):entry(name,'scan',28,[],episode=e)
    entry('selection','selection',0,scans)
    fit_deps=['selection']
    for e in range(128):
        verification=f'verify_{e:03d}'
        entry(verification,'verify',196,['selection',scans[e]],episode=e)
        fit_deps.append(verification)
        for rank in (0,1):
            for g in range(8):
                name=f'group_{e:03d}_{rank}_{g}'
                entry(name,'group',224,['selection',scans[e],verification],episode=e,rank=rank,group=g)
                fit_deps.append(name)
    entry('fit','fit',0,fit_deps)
    assessments=[f'assess_{e:03d}' for e in range(128)]
    for e,name in enumerate(assessments):entry(name,'assessment',112,['fit'],episode=e)
    entry('replays','replays',112,['fit','assess_000'])
    entry('summary','summary',0,assessments+['replays'])
    assert len(plan['units'])==len(specs)==2436
    for unit,(spec,maximum) in zip(plan['units'],specs):
        expected=dict(item=spec['item'],specification=spec,days=maximum,
                      phase='assessment' if spec['kind'] in ('assessment','replays','summary') else 'fit',
                      seconds=study['seconds'][spec['kind']],
                      input_identity=hashed(dict(study=hashed(study),specification=spec)))
        assert unit==expected
    assert sum(u['days'] for u in plan['units'])==501872


def selection(observations,episode,base,namespace):
    assert set(observations)=={1,8,15,22}
    candidates=[(d,i) for d in (1,8,15,22) for i in range(10)]
    scores=[]
    for day,sku in candidates:
        x=np.asarray(observations[day],float)
        deficits=np.maximum(7*x[-20:-10]-x[:10],0)
        excess=max(x[sku]-7*x[-20+sku],0)
        scores.append(excess/500*((deficits.sum()-deficits[sku])/500)*(1-np.clip(x[-10+sku]/500,0,1)))
    best=max(range(40),key=lambda i:scores[i])
    remaining=[i for i in range(40) if i!=best]
    chosen=int(np.random.default_rng(np.random.SeedSequence([namespace,episode,991])).integers(39))
    return [list(candidates[best]),list(candidates[remaining[chosen]])]


def check_models(samples,base,policy):
    assert [(s['episode'],s['rank']) for s in samples]==[(e,r) for e in range(128) for r in (0,1)]
    assert all(len(s['gains'])==32 for s in samples)
    assert policy['samples']==samples
    np.testing.assert_array_equal(policy['base_prices'],base)
    targets=np.array([float(np.mean(s['gains'])/base[s['sku']]) for s in samples])
    np.testing.assert_array_equal(policy['targets'],targets)
    assert set(policy['models'])==set(policy['matrices'])=={'local','cross'}
    for arm in ('local','cross'):
        raw=np.array([feature_matrix(s['obs'],s['day'],base,arm)[s['sku']] for s in samples])
        np.testing.assert_allclose(policy['matrices'][arm],raw,rtol=0,atol=1e-14)
        assert policy['models'][arm]['arm']==arm
        check_ridge(raw,targets,policy['models'][arm])


def check_state_groups(original,state,groups,episode,rank,namespaces,covers,base):
    """Reconstruct one target from all32 futures and audit each full suffix.

Returns independent sample and counted steps (groups plus probe/two replays).
The caller separately audits the original scan/verification full-window walks.
"""
    day,sku=state['day'],state['sku']
    before=state['before']
    assert state['rank']==rank and day in (1,8,15,22) and 0<=sku<10
    assert before['day']==day and before['pending'] is None
    assert before['start']==original[0]['start']
    assert state['obs']==original[day-1]['pricing_obs']
    assert state['probe']==original[day-1:]
    np.testing.assert_array_equal(np.array(before['inventory'],np.float32),np.array(before['obs'],np.float32)[:10])
    np.testing.assert_array_equal(np.array(before['history'],np.float32).T.ravel(),np.array(before['obs'])[80:220])
    original_keys=original_noise_keys(namespaces['scan']+episode)
    audit_trace(state['probe'],day,before['start'],original_keys,covers,base,{})
    assert len(state['replays'])==2 and len(groups)==8
    count=29-day
    gains=[]
    for group_index,group in enumerate(groups):
        assert (group['episode'],group['rank'],group['group'])==(episode,rank,group_index)
        assert (group['key'],group['day'],group['sku'])==(namespaces['scan']+episode,day,sku)
        assert group['before']==before and group['prefix']==original[:day-1]
        count+=len(group['prefix'])
        assert len(group['draws'])==4
        for offset,draw in enumerate(group['draws']):
            replica=4*group_index+offset
            assert draw['replica']==replica
            keys=[int(np.random.SeedSequence([namespaces['future'],episode,day,sku,replica,z]).generate_state(1,dtype=np.uint64)[0]) for z in (0,1)]
            assert draw['keys']==keys and len(draw['branches'])==2
            values=[]
            for position,action in enumerate((0,4)):
                branch=draw['branches'][position]
                assert branch['action']==action and branch['initial']==before
                first=branch['trace'][0]
                assert first['pricing_obs']==state['obs']
                for section in (slice(0,220),slice(230,250)):
                    np.testing.assert_array_equal(np.array(first['obs'])[section],np.array(before['obs'])[section])
                np.testing.assert_array_equal(first['info']['orders_delivered'],before['accepted'])
                np.testing.assert_array_equal(first['info']['orders_clipped'],before['clipped'])
                value=audit_trace(branch['trace'],day,before['start'],keys,covers,base,{},sku,action)
                assert value==branch['value']
                count+=len(branch['trace'])
                values.append(value)
                if replica==0:
                    assert state['keys']==keys and state['replays'][position]==branch['trace']
                    audit_trace(state['replays'][position],day,before['start'],keys,covers,base,{},sku,action)
                    count+=len(state['replays'][position])
            gains.append(values[0]-values[1])
    assert count==8*(day-1)+67*(29-day)
    return dict(episode=episode,rank=rank,day=day,sku=sku,obs=state['obs'],gains=gains),count


def check_analysis(rows,summary,namespaces):
    labels=('cross','local','baseline','heuristic')
    keys=list(range(namespaces['assessment'],namespaces['assessment']+128))
    table={(r['label'],r['key']):r for r in rows}
    assert len(table)==len(rows)==512 and set(table)=={(l,k) for l in labels for k in keys}
    draws=np.random.default_rng(namespaces['bootstrap']).integers(128,size=(10000,128))
    effects={}
    for ref in ('baseline','local','heuristic'):
        for metric in ('reward_day','fill'):
            delta=np.array([table['cross',k][metric]-table[ref,k][metric] for k in keys])
            assert np.isfinite(delta).all()
            bounds=np.quantile([delta[d].mean() for d in draws],[.05/12,1-.05/12])
            stored=summary['effects'][f'{ref}:{metric}']
            assert stored['mean']==float(delta.mean())
            np.testing.assert_allclose(stored['ci991667'],bounds,rtol=0,atol=1e-12)
            effects[ref,metric]=(float(delta.mean()),bounds)
    assert len(summary['effects'])==6
    for ref,gate in [('baseline','primary_pass'),('local','representation_pass'),('heuristic','heuristic_pass')]:
        mean,bounds=effects[ref,'reward_day']
        passed=bounds[0]>0 and effects[ref,'fill'][1][0]>=-.005 and (ref!='baseline' or mean>=.5)
        assert summary[gate]==bool(passed)
    for label in labels:
        for metric in ('reward_day','fill'):
            assert summary['means'][label][metric]==float(np.mean([table[label,k][metric] for k in keys]))
        np.testing.assert_array_equal(summary['price_counts'][label],np.sum([table[label,k]['price_counts'] for k in keys],axis=0))


def verify_archive(root,study,plan):
    check_plan(study,plan)
    structure=check_archive(root,study,plan,require_complete=True)
    root=Path(root).resolve()
    db=sqlite3.connect((root/'attempts.sqlite').as_uri()+'?mode=ro',uri=True)
    try:
        entries=db.execute("SELECT unit,artifact,days FROM attempts WHERE status='complete' ORDER BY id").fetchall()
    finally:
        db.close()
    index={plan['units'][i]['item']:(root/path,days) for i,path,days in entries}
    audited=set()
    def load(item,expected_days):
        path,days=index[item]
        assert days==expected_days
        record=json.loads(path.read_text(encoding='utf-8'))
        assert record['days']==days
        audited.add(item)
        return record['payload']
    calibration=json.loads(Path(study['inputs']['calibration']['path']).read_text())
    prices={sku.series_id:sku.base_price for sku in load_skus()}
    base=np.array([prices[s] for s in calibration['series_order']])
    covers=np.array(json.loads(Path(study['inputs']['ordering']['path']).read_text())['mean'],np.float32)
    ns=study['namespaces']
    frozen=load('selection',0)
    assert len(frozen['selections'])==128
    samples=[];length=0;successful_days=0
    for ep in range(128):
        key=ns['scan']+ep
        original=load(f'scan_{ep:03d}',28)
        assert original['episode']==ep and original['key']==key
        trace=original['trace']
        start=int(np.random.default_rng(np.random.SeedSequence([key,314159])).integers(15,1859))
        audit_trace(trace,1,start,original_noise_keys(key),covers,base,{})
        obs={r['day']:r['pricing_obs'] for r in trace if r['pricing_obs'] is not None}
        chosen=selection(obs,ep,base,ns['scan'])
        selected=frozen['selections'][ep]
        assert selected['episode']==ep and selected['key']==key
        assert original['selected']==selected['selected']==chosen
        for record in (original,selected):
            assert {int(k):v for k,v in record['observations'].items()}==obs
        local_length=sum(29-d for d,_ in chosen);length+=local_length
        verification=load(f'verify_{ep:03d}',28+3*local_length)
        assert verification['episode']==ep and verification['key']==key
        assert verification['original_replay']==trace and len(verification['states'])==2
        # Same trace has already been checked, but it represents a second execution.
        successful_days+=56
        for rank in (0,1):
            state=verification['states'][rank]
            assert [state['day'],state['sku']]==chosen[rank]
            day=state['day']
            groups=[load(f'group_{ep:03d}_{rank}_{g}',day-1+8*(29-day)) for g in range(8)]
            sample,count=check_state_groups(trace,state,groups,ep,rank,ns,covers,base)
            samples.append(sample);successful_days+=count
    assert frozen['suffix_length_sum']==length
    assert frozen['successful_days']==78960+59*length
    policy=load('fit',0)
    check_models(samples,base,policy)
    models=policy['models'];rows=[];first=None
    labels=('cross','local','baseline','heuristic')
    for ep in range(128):
        assessment=load(f'assess_{ep:03d}',112)
        assert assessment['episode']==ep
        assert [r['label'] for r in assessment['records']]==list(labels)
        key=ns['assessment']+ep
        start=int(np.random.default_rng(np.random.SeedSequence([key,314159])).integers(15,1859))
        for record in assessment['records']:
            assert record['key']==key
            label=record['label'];trace=record['trace']
            audit_trace(trace,1,start,original_noise_keys(key),covers,base,models,label=label)
            demand=sum(sum(r['info']['demand']) for r in trace)
            result=dict(key=key,label=label,reward_day=sum(r['reward'] for r in trace)/28,
                fill=sum(sum(r['info']['sales']) for r in trace)/demand if demand else 1.,
                price_counts=np.bincount(np.concatenate([r['price_actions'] for r in trace if r['price_actions'] is not None]),minlength=5).tolist())
            assert record['result']==result
            rows.append(result)
        successful_days+=112
        if ep==0:first=assessment
    assert load('replays',112)==first
    successful_days+=112
    summary=load('summary',0)
    check_analysis(rows,summary,ns)
    assert successful_days==frozen['successful_days']==sum(days for _,_,days in entries)
    assert audited==set(index) and len(audited)==2436
    return dict(status='passed',scientific_verification=True,units=2436,
                successful_days=successful_days,charged_days=structure['charged']['days'],
                new_simulator_days=0,states=256,assessment_episodes=512,
                known_elapsed_seconds=structure['known_elapsed_seconds'],
                unknown_elapsed_attempts=structure['unknown_elapsed_attempts'],
                primary_pass=summary['primary_pass'],representation_pass=summary['representation_pass'],
                heuristic_pass=summary['heuristic_pass'])
