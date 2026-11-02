"""Complete synthetic disk archive; no real environment or research samples."""
import hashlib
import json
from types import SimpleNamespace
import sqlite3
import numpy as np
import pytest
from scripts import recovery_campaign_science as producer
from scripts import verify_recovery_science as verifier
from scripts import verify_recovery_campaign as final_audit
from scripts.recovery_campaign_plan import build
from scripts.check_recovery_campaign_archive import check
from src.experiments.execution_journal import digest
from src.experiments.recovery_ledger import RecoveryLedger
from tests.test_recovery_campaign_plan import study


def test_complete_disk_science_and_rehashed_target_tampering(tmp_path,monkeypatch):
    root=tmp_path/'archive';root.mkdir()
    config=study()
    calibration=tmp_path/'calibration.json';ordering=tmp_path/'ordering.json'
    calibration.write_text(json.dumps(dict(series_order=list(range(10)))))
    ordering.write_text(json.dumps(dict(mean=[0]*10)))
    config['inputs']={k:dict(path=str(p),sha256=digest(p)) for k,p in [('calibration',calibration),('ordering',ordering)]}
    config['source_hashes']={str(p):digest(p) for p in (calibration,ordering)}
    config['callback']=dict(module='scripts.recovery_campaign_dispatch',function='dispatch')
    plan=build(config);ns=config['namespaces'];base=np.ones(10)
    monkeypatch.setattr(verifier,'load_skus',lambda:[SimpleNamespace(series_id=i,base_price=1.) for i in range(10)])
    ledger=RecoveryLedger(root/'attempts.sqlite',**plan);ledger.close()
    zero=[0]*10;obs=[0]*260;pobs=[0]*204
    pobs[150:160]=[1.2]*10;pobs[-20:-10]=[1]*10
    records={};scans=[];samples=[];assessments=[];verification=None
    def trace(key,first=1,sku=0,action=4,noise=None):
        start=int(np.random.default_rng(np.random.SeedSequence([key,314159])).integers(15,1859))
        keys=verifier.original_noise_keys(key) if noise is None else noise
        rows=[]
        for day in range(first,29):
            pa=None
            if (day-1)%7==0:
                pa=[4]*10
                if day==first:pa[sku]=action
                prices=np.array([.8,.9,1.,1.1,1.2])[pa].tolist()
            rows.append(dict(day=day,start=start,demand_seed=keys[0],supplier_seed=keys[1],obs=obs,actions=zero,
                pricing_obs=pobs if pa is not None else None,price_actions=pa,reward=0.,
                info=dict(price=prices,total_on_hand=0.,orders_delivered=zero,orders_clipped=zero,
                demand=[1]*10,sales=[1]*10,reward_terms=dict(revenue=0.,ordering_cost=0.,holding_cost=0.,stockout_cost=0.,price_change_cost=0.))))
        return rows
    with sqlite3.connect(root/'attempts.sqlite') as db:
        for index,unit in enumerate(plan['units']):
            spec=unit['specification'];kind=spec['kind'];ep=spec.get('episode',0)
            if kind=='scan':
                key=ns['scan']+ep;t=trace(key)
                observations={r['day']:r['pricing_obs'] for r in t if r['pricing_obs'] is not None}
                payload=dict(episode=ep,key=key,trace=t,observations=observations,
                    selected=producer.select(observations,ep,base,ns['scan']))
                scans.append(payload);days=28
            elif kind=='selection':
                frozen=producer.freeze(scans,base,ns);payload=frozen;days=0
            elif kind=='verify':
                original=scans[ep];key=original['key'];states=[]
                for rank,(day,sku) in enumerate(original['selected']):
                    before=dict(day=day,pending=None,start=original['trace'][0]['start'],inventory=zero,
                                history=[[0]*10 for _ in range(14)],obs=obs,accepted=zero,clipped=zero)
                    keys=producer.keys(ns['future'],ep,day,sku,0)
                    states.append(dict(rank=rank,day=day,sku=sku,before=before,obs=pobs,
                        probe=original['trace'][day-1:],keys=keys,replays=[trace(key,day,sku,a,keys) for a in (0,4)]))
                    samples.append(dict(episode=ep,rank=rank,day=day,sku=sku,obs=pobs,gains=[0.]*32))
                verification=dict(episode=ep,key=key,states=states,original_replay=original['trace'])
                payload=verification;days=28+3*sum(29-s['day'] for s in states)
            elif kind=='group':
                state=verification['states'][spec['rank']];day=state['day'];sku=state['sku'];key=ns['scan']+ep
                draws=[]
                for rep in range(spec['group']*4,spec['group']*4+4):
                    keys=producer.keys(ns['future'],ep,day,sku,rep)
                    draws.append(dict(replica=rep,keys=keys,branches=[dict(action=a,initial=state['before'],
                        trace=trace(key,day,sku,a,keys),value=0.) for a in (0,4)]))
                payload=dict(episode=ep,rank=spec['rank'],group=spec['group'],key=key,day=day,sku=sku,
                             before=state['before'],prefix=scans[ep]['trace'][:day-1],draws=draws)
                days=day-1+8*(29-day)
            elif kind=='fit':payload=producer.fit_samples(samples,base);days=0
            elif kind=='assessment':
                key=ns['assessment']+ep;t=trace(key)
                payload=dict(episode=ep,records=[dict(label=l,key=key,trace=t,result=dict(label=l,key=key,
                    reward_day=0.,fill=1.,price_counts=[0,0,0,0,40])) for l in ('cross','local','baseline','heuristic')])
                assessments.append(payload);days=112
            elif kind=='replays':payload=assessments[0];days=112
            else:payload=producer.analyze(assessments,ns);days=0
            arguments=dict(study=config,unit=unit,dependencies=[records[name] for name in spec['dependencies']])
            request=dict(sequence=index+1,module=config['callback']['module'],function='dispatch',arguments=arguments,
                         maximum_days=unit['days'],deadline=index*2+120.,dispatched=index*2.)
            request_hash=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
            receipt=dict(sequence=index+1,request_sha256=request_hash,request=request,days=days,
                         started=index*2+.01,ended=index*2+.1,received=index*2+.11)
            artifact=dict(item=unit['item'],input_identity=unit['input_identity'],days=days,payload=payload,worker_receipt=receipt)
            path=root/f'attempt_{index+1:06d}.json';path.write_text(json.dumps(artifact))
            checksum=digest(path)
            db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?,?)',(index+1,index,'complete',days,.2,path.name,checksum))
            records[unit['item']]=dict(item=unit['item'],path=str(path.resolve()),sha256=checksum,input_identity=unit['input_identity'])
    # Real invocation/accounting/science checks; host source/runtime preflight
    # is replaced because this fixture deliberately uses synthetic SKU inputs.
    control=root.with_name(root.name+'_invocations');control.mkdir()
    fit_end=next(i for i,u in enumerate(plan['units']) if u['item']=='fit')+1
    declaration=dict(study=config,protocol='synthetic-only',runtime={},
        schedule=[[0,fit_end],[fit_end,len(plan['units'])]],
        overhead=dict(per_invocation_seconds=2.,maximum_invocations=3,fit=5.,assessment=5.,total=10.))
    (control/'declaration.json').write_text(json.dumps(declaration))
    for number,(start,end) in enumerate(declaration['schedule'],1):
        (control/f'invocation_{number:06d}.json').write_text(json.dumps(dict(number=number,status='complete',
            attempts_before=start,attempts_after=end,completed_before=start,completed_after=end,
            span=[start,end],elapsed_seconds=(end-start)*.2+.1,trailing_seconds=.1)))
    monkeypatch.setattr(final_audit,'validate_coverage',lambda *args:None)
    monkeypatch.setattr(final_audit,'runtime_identity',lambda:{})
    combined=final_audit.verify(root,declaration,plan)
    assert combined['archive_verified'] and combined['execution']['combined_budget_checked']
    assert not combined['full_project_goal_achieved']
    result=combined['science']
    assert result['status']=='passed' and result['new_simulator_days']==0
    assert result['successful_days']==frozen['successful_days']
    assert not any(result[k] for k in ('primary_pass','representation_pass','heuristic_pass'))
    # Rehash every downstream dependency/request/ledger link. Structure passes;
    # the independently reconstructed scientific targets must still reject it.
    records={}
    with sqlite3.connect(root/'attempts.sqlite') as db:
        for index,unit in enumerate(plan['units']):
            path=root/f'attempt_{index+1:06d}.json';artifact=json.loads(path.read_text())
            if unit['item']=='fit':artifact['payload']['targets'][0]=1.
            request=artifact['worker_receipt']['request']
            request['arguments']['dependencies']=[records[name] for name in unit['specification']['dependencies']]
            artifact['worker_receipt']['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
            path.write_text(json.dumps(artifact));checksum=digest(path)
            db.execute('UPDATE attempts SET sha256=? WHERE id=?',(checksum,index+1))
            records[unit['item']]=dict(item=unit['item'],path=str(path.resolve()),sha256=checksum,input_identity=unit['input_identity'])
    assert check(root,config,plan)['scientific_verification'] is False
    with pytest.raises(AssertionError):final_audit.verify(root,declaration,plan)
    assert not (root/'verification.json').exists()
