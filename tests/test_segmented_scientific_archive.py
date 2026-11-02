"""Complete synthetic archive: no environment, demand sampling or research fit."""
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
from scripts import cross_sku_segmented_units as u
from scripts import verify_cross_sku_segmented as v
from scripts.check_segmented_archive import sha256


def test_complete_scientific_archive(tmp_path,monkeypatch):
    root=tmp_path/'archive';root.mkdir();entries=[];scans=[];branches=[]
    base=np.ones(10);zero=[0]*10;obs=[0]*260
    pobs=[0]*204;pobs[150:160]=[1]*10;pobs[-20:-10]=[1]*10
    calibration=tmp_path/'train_only_nb_identifiable_v2.json'
    checkpoint=tmp_path/'sku_seed211_final.json'
    calibration.write_text(json.dumps(dict(series_order=list(range(10)))))
    checkpoint.write_text(json.dumps(dict(mean=zero)))
    monkeypatch.setattr(v,'load_skus',lambda:[SimpleNamespace(series_id=i,base_price=1.) for i in range(10)])
    hashes={str(p):sha256(p) for p in [calibration,checkpoint]}
    caps=dict(fit=7200.,assessment=600.,total=7800.)
    manifest=dict(items=u.plan(),caps=caps,maximum_days=501872,hashes=hashes,covers=zero)
    def save(item,payload,days):
        name=f'unit_{len(entries):04d}.json';path=root/name
        path.write_text(json.dumps(dict(item=item,days=days,payload=payload),separators=(',',':')))
        entries.append(dict(item=item,days=days,artifact=name,sha256=sha256(path)))
    def trace(key,first=1,sku=-1,action=4,noise=None):
        start=int(np.random.default_rng(np.random.SeedSequence([key,314159])).integers(15,1859))
        keys=v.original_noise_keys(key) if noise is None else noise;rows=[]
        for day in range(first,29):
            pa=None
            if (day-1)%7==0:
                pa=[4]*10
                if day==first and sku>=0:pa[sku]=action
                prices=np.array([.8,.9,1.,1.1,1.2])[pa].tolist()
            rows.append(dict(day=day,start=start,demand_seed=keys[0],supplier_seed=keys[1],
                obs=obs,actions=zero,pricing_obs=pobs if pa is not None else None,price_actions=pa,reward=0.,
                info=dict(price=prices,total_on_hand=0.,orders_delivered=zero,orders_clipped=zero,
                    demand=[1]*10,sales=[1]*10,reward_terms=dict(revenue=0.,ordering_cost=0.,
                    holding_cost=0.,stockout_cost=0.,price_change_cost=0.))))
        return rows
    for ep in range(128):
        t=trace(2570000+ep);observations={r['day']:r['pricing_obs'] for r in t if r['pricing_obs'] is not None}
        scan=dict(episode=ep,key=2570000+ep,trace=t,observations=observations,
            selected=u.select(observations,ep,base),base_prices=base.tolist())
        scans.append(scan);save(f'scan_{ep:03d}',scan,28)
    selected,_=u.freeze(scans,base,[r['sha256'] for r in entries],hashes);save('selection',selected,0)
    for ep in range(128):
        states=[];key=2570000+ep
        for rank,(day,sku) in enumerate(selected['selections'][ep]['selected']):
            before=dict(day=day,pending=None,start=scans[ep]['trace'][0]['start'],inventory=zero,
                history=[[0]*10 for _ in range(14)],obs=obs,accepted=zero,clipped=zero)
            draws=[]
            for rep in range(32):
                keys=[int(np.random.SeedSequence([2580000,ep,day,sku,rep,z]).generate_state(1,dtype=np.uint64)[0]) for z in [0,1]]
                draws.append(dict(keys=keys,branches=[dict(action=a,initial=before,
                    trace=trace(key,day,sku,a,keys),value=0.) for a in [0,4]]))
            sample=dict(episode=ep,rank=rank,day=day,sku=sku,obs=pobs,gains=[0.]*32,target=0.,
                features={arm:u.matrix(pobs,day,base,arm)[sku].tolist() for arm in ['local','cross']})
            states.append(dict(sample=sample,before=before,after=before,probe=scans[ep]['trace'][day-1:],
                draws=draws,replays=[b['trace'] for b in draws[0]['branches']]))
        payload=dict(episode=ep,key=key,original_replay=scans[ep]['trace'],states=states)
        save(f'branch_{ep:03d}',payload,28+67*sum(29-d for d,s in selected['selections'][ep]['selected']))
        branches.append(dict(episode=ep,states=[dict(sample=s['sample']) for s in states]))
    policy,_=u.fit_models(branches,base,entries[128]['sha256'],[r['sha256'] for r in entries[129:257]],hashes)
    save('fit',policy,0);assessment=[]
    for ep in range(128):
        key=2590000+ep;t=trace(key)
        records=[dict(label=l,key=key,trace=t,result=dict(label=l,key=key,reward_day=0.,fill=1.,
            price_counts=[0,0,0,0,40])) for l in u.LABELS]
        a=dict(episode=ep,records=records);assessment.append(a);save(f'assess_{ep:03d}',a,112)
    save('replays',assessment[0],112)
    total=selected['planned_days']
    save('summary',dict(status='complete',analysis=u.analyze(assessment),days=total,
        suffix_length_sum=selected['suffix_length_sum'],test_scoring=False),0)
    identity=hashlib.sha256(json.dumps(manifest,sort_keys=True,allow_nan=False).encode()).hexdigest()
    history=[dict(item=item,phase='fit' if i<=257 else 'assessment',elapsed=1.,start_wall=float(i*2),
        end_wall=float(i*2+1),boundary_gap_seconds=None if i==0 else 1.) for i,item in enumerate(u.plan())]
    for name,payload in [('manifest.json',manifest),('journal.json',dict(
            plan=dict(identity=identity,items=u.plan(),maximum_days=501872),pending=None,completed=entries,reserved_days=total)),
            ('time.json',dict(plan=dict(identity=identity,caps=caps),status='idle',pending=None,
                completed=u.plan(),used=dict(fit=258.,assessment=130.,total=388.),history=history))]:
        (root/name).write_text(json.dumps(payload))
    v.verify(root)
    result=json.loads((root/'verification.json').read_text())
    assert result['status']=='passed' and result['days']==total and result['new_simulator_days']==0
    assert not any(result[k] for k in ['primary_pass','representation_pass','heuristic_pass'])
    # Rehash the altered branch and its model-freeze reference: structural
    # validation must pass, then scientific reconstruction must reject it.
    path=root/'unit_0129.json';changed=json.loads(path.read_text())
    changed['payload']['states'][0]['sample']['target']=1.
    path.write_text(json.dumps(changed,separators=(',',':')));entries[129]['sha256']=sha256(path)
    path=root/'unit_0257.json';changed=json.loads(path.read_text())
    changed['payload']['branch_hashes'][0]=entries[129]['sha256']
    path.write_text(json.dumps(changed,separators=(',',':')));entries[257]['sha256']=sha256(path)
    path=root/'journal.json';changed=json.loads(path.read_text());changed['completed']=entries
    path.write_text(json.dumps(changed))
    assert v.check_archive(root)['scientific_verification'] is False
    (root/'verification.json').unlink()
    with pytest.raises(AssertionError):v.verify(root)
    assert not (root/'verification.json').exists()
