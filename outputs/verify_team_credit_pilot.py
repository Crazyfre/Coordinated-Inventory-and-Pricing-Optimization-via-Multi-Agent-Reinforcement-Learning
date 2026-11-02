"""Original-supplier replay of matched whole-team/integrated credit packages.

No SGD refit. Checks all preserved prebatch policies, stochastic actual paths,
nonidentity branches, identity proofs/selfchecks, greedy paths and inference.
"""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from outputs.team_credit_pilot_support import (ROOT,ARMS,TRAIN_KEYS,ASSESS_KEYS,SEED,TORCH_SEED,UPDATES,BATCH,
    MAIN_DAYS,AUDIT_DAYS,MAIN_INTERNAL_SECONDS,MAIN_EXTERNAL_SECONDS,AUDIT_INTERNAL_SECONDS,AUDIT_EXTERNAL_SECONDS,
    CALIBRATION,COVERS,Accounting,policy,calendar,load)
from outputs.conditional_credit_pilot_replay import replay_training,replay_greedy,serial
from outputs.support_pilot_replay import replay_integrated
from src.algos.parameter_policy_gradient import order_action
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env
from src.experiments.execution_journal import atomic_json,digest
from scripts.recovery_source_inventory import runtime_identity

OUT=ROOT.with_name(ROOT.name+'_audit')
INTERNAL_SECONDS=AUDIT_INTERNAL_SECONDS
EXTERNAL_SECONDS=AUDIT_EXTERNAL_SECONDS


def independent_analysis(records):
    indices=np.random.default_rng(20280113).integers(96,size=(20000,96));effects={};means={}
    for arm,rows in records.items():means[arm]={m:float(np.average([r[m] for r in rows])) for m in ('reward_day','fill')}
    for treatment,control in (('integrated','anchor'),('whole','anchor'),('whole','integrated')):
        contrast={}
        for metric in ('reward_day','fill'):
            values=np.subtract([r[metric] for r in records[treatment]],[r[metric] for r in records[control]])
            averages=values[indices].sum(axis=1)/96
            contrast[metric]=dict(mean=float(np.average(values)),ci=np.percentile(averages,[100*.05/12,100*(1-.05/12)]).tolist())
        contrast['gate']=bool(contrast['reward_day']['ci'][0]>0 and contrast['fill']['ci'][0]>=-.005 and
            (control!='anchor' or contrast['reward_day']['mean']>=.5))
        effects[treatment+':'+control]=contrast
    learning={arm:effects[arm+':anchor']['gate'] for arm in ('integrated','whole')}
    return dict(means=means,effects=effects,learning_gates=learning,
        qualifying_arms=[arm for arm in ('integrated','whole') if learning[arm]],
        any_learning_gate=any(learning.values()),all_learning_gates=all(learning.values()),
        peer_superiority_gate=effects['whole:integrated']['gate'],confidence=1-.05/6,
        bootstrap_replicates=20000,bootstrap_seed=20280113,trained_seeds=1,
        uncertainty='paired scenario resampling conditional on one trained seed',full_goal_achieved=False)


def equal(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(str,bool)) or a is None:assert a==b
    elif isinstance(a,(list,tuple)):
        assert isinstance(b,(list,tuple)) and len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert np.allclose(a,b,atol=1e-10,rtol=1e-9)


def measure(ep,covers):
    trace=ep['trace'];demand=np.stack([r['info']['demand'] for r in trace]);sales=np.stack([r['info']['sales'] for r in trace])
    fill=float(sales.sum()/max(demand.sum(),1e-12));prices=np.hstack([r['price_actions'] for r in trace if r['price_actions'] is not None])
    return dict(reward_day=float(np.sum(ep['rewards'])/28),episode_return=float(np.sum(ep['rewards'])),fill=fill,
        fill_by_sku=np.divide(sales.sum(0),np.maximum(demand.sum(0),1e-12)).tolist(),absolute_service=bool(fill>=.95),
        order_changes=sum(int(np.count_nonzero(np.asarray(r['actions'])!=order_action(np.asarray(r['obs']),r['day'],covers))) for r in trace),
        price_changes=int(np.sum(prices!=4)),price_indices=sorted(set(int(v) for v in prices)),lost_units=float(np.sum(demand-sales)),
        stockout_sku_days=int(np.sum(demand>sales+1e-8)))


def optimizer_transition(before,after,steps,lr):
    assert len(before['param_groups'])==len(after['param_groups'])==1
    assert before['param_groups']==after['param_groups']
    assert before['param_groups'][0]['lr']==lr
    ids=after['param_groups'][0]['params']
    if steps==0:assert before['state'].keys()==after['state'].keys()
    for key in ids:
        old=before['state'].get(key);new=after['state'].get(key)
        if old is None and steps==0:assert new is None;continue
        assert new is not None and float(new['step'])==(float(old['step']) if old else 0)+steps
        assert all(torch.isfinite(v).all() for v in new.values() if isinstance(v,torch.Tensor))
        if steps==0:assert all(torch.equal(v,new[k]) for k,v in old.items())


def update_transition(model,arm,before,after,entry,targets):
    assert entry['metrics']['value_norm_updates']==int(after['model']['value_norm.updates'])==int(before['model']['value_norm.updates'])+1
    for role in ('pricing','ordering'):
        steps=entry['metrics'][role+'_actor_steps'];assert isinstance(steps,int) and 0<=steps<=4
        optimizer_transition(before['actors'][role],after['actors'][role],steps,.003)
        expect=float(torch.tensor(targets[role],dtype=torch.float32).mean())
        assert np.isclose(expect,entry['metrics'][role+'_critic_target_mean'],atol=3e-6,rtol=5e-5)
    optimizer_transition(before['critic_optimizer'],after['critic_optimizer'],4,.001)
    t=[np.asarray(targets[r],dtype=np.float32).astype(np.float64) for r in ('pricing','ordering')]
    mean=np.mean([x.mean() for x in t]);square=np.mean([np.square(x).mean() for x in t])
    for key,value in (('mean_ema',mean),('square_ema',square),('mass',1.)):
        expected=float(before['model']['value_norm.'+key])*.99999+value*1e-5
        assert np.isclose(float(after['model']['value_norm.'+key]),expected,rtol=1e-9,atol=2e-12)
    for role in ('pricing','ordering','critics'):
        changed=any(not torch.equal(v,after['model'][k]) for k,v in before['model'].items() if k.startswith(role+'.'))
        assert changed==entry['changes'][role]
    assert set(before['model'])==set(after['model'])
    assert all(torch.isfinite(v).all() for v in after['model'].values())
    # Schema, local actor buffers and optimizer state must load as declared.
    model.restore(after)


def main():
    OUT.mkdir(exist_ok=False);began=time.perf_counter();deadline=began+INTERNAL_SECONDS
    torch.set_num_threads(1);budget=Accounting(OUT/'step_ledger.jsonl',AUDIT_DAYS,deadline)
    stage='preflight';main_steps=0;main_ledger=None;assessment_ledger=None;stream_counts={}
    def verify_main(kind,context,days,stream):
        nonlocal main_steps
        for _ in range(days):
            stream_counts[id(stream)]=stream_counts.get(id(stream),0)+1
            claim=json.loads(next(stream));done=json.loads(next(stream))
            expected=dict(step=stream_counts[id(stream)],kind=kind,**context)
            assert claim==dict(event='claim',**expected) and done==dict(event='complete',**expected)
            main_steps+=1
    try:
        manifest=json.loads((ROOT/'manifest.json').read_text());freeze=json.loads((ROOT/'training_freeze.json').read_text())
        assessment=json.loads((ROOT/'assessment_freeze.json').read_text());report=json.loads((ROOT/'report.json').read_text())
        reference_freeze=json.loads((ROOT/'reference_freeze.json').read_text())
        assert manifest['runtime']==runtime_identity() and manifest['seed']==SEED and manifest['torch_seed']==TORCH_SEED
        assert manifest['train_keys']==list(TRAIN_KEYS) and manifest['assessment_keys']==list(ASSESS_KEYS)
        assert manifest['arms']==list(ARMS) and manifest['updates']==UPDATES and manifest['batch']==BATCH
        assert manifest['maximum_days']==MAIN_DAYS and manifest['internal_seconds']==MAIN_INTERNAL_SECONDS and manifest['external_seconds']==MAIN_EXTERNAL_SECONDS
        assert manifest['backend']=='vectorized_day_sku_inverse_cdf' and manifest['assessment_split']=='train'
        assert report['status']=='complete' and report['days']==report['reserved_days']<=MAIN_DAYS
        assert report['updates']==freeze['updates']==64 and report['training_episodes']==2048 and report['assessment_episodes']==288
        assert len(freeze['checkpoints'])==67 and len(freeze['data'])==1089
        expected_checkpoints={'baseline.pt'}|{f'{a}/batch_{b:03d}.pt' for a in ARMS for b in range(32)}|{f'{a}/final.pt' for a in ARMS}
        assert set(freeze['checkpoints'])==expected_checkpoints
        assert set(freeze['data'])=={'training_ledger.jsonl'}|{f'references/{k}.pt' for k in TRAIN_KEYS}|{f'{a}/data_{b:03d}.pt' for a in ARMS for b in range(32)}
        assert set(assessment['hashes'])=={'assessment_ledger.jsonl'}|{f'assessment_{a}.pt' for a in ('anchor',)+ARMS}
        assert reference_freeze['hashes']=={p:h for p,h in freeze['data'].items() if p.startswith('references/')}
        assert reference_freeze['completed_wall']<=freeze['completed_wall']
        assert all(digest(p)==h for p,h in manifest['hashes'].items())
        allfiles={**freeze['checkpoints'],**freeze['data'],**assessment['hashes']}
        assert all(digest(ROOT/p)==h for p,h in allfiles.items())
        start=json.loads((ROOT/'assessment_start.json').read_text())
        assert start['training_freeze']==digest(ROOT/'training_freeze.json') and start['started_wall']>=freeze['completed_wall']
        params=load_identifiable_training_calibration(CALIBRATION)
        env=make_research_env('S2',split='train',start=15,days=28,pricing_scope='per_sku',params=params)
        covers=np.asarray(json.loads(Path(COVERS).read_text())['mean']);anchor=policy('anchor',env,covers)
        baseline=load(ROOT/'baseline.pt')
        assert all(torch.equal(v,baseline['model'][k]) for k,v in anchor.state_dict().items())
        assert all(not d['state'] for d in baseline['actors'].values()) and not baseline['critic_optimizer']['state']
        main_ledger=iter((ROOT/'training_ledger.jsonl').read_text().splitlines())
        assessment_ledger=iter((ROOT/'assessment_ledger.jsonl').read_text().splitlines())
        references={};stage='references'
        for key in TRAIN_KEYS:
            calendar(env,key);budget.context=dict(arm='anchor',batch=None,key=key)
            ep=replay_greedy(anchor,env,key,budget.advance,'reference');saved=load(ROOT/f'references/{key}.pt')
            assert saved['key']==key and saved['rewards']==ep['rewards'] and serial(saved['trace'])==serial(ep['trace'])
            references[key]=ep['rewards']
            verify_main('reference',dict(arm='anchor',batch=None,key=key,root_role=None,root_day=None),28,main_ledger)
        totals=dict(branch_days=0,skips=0,selfchecks=0,common_pairs=0,same_price_pairs=0);counts={};stage='training'
        initial_new=None
        for arm in ARMS:
            model=policy(arm,env,covers);counts[arm]={n:sum(p.numel() for p in getattr(model,n).parameters()) for n in ('pricing','ordering','critics')}
            initial=load(ROOT/f'{arm}/batch_000.pt');torch.manual_seed(TORCH_SEED)
            assert torch.equal(torch.get_rng_state(),initial['rng']) and all(torch.equal(v,initial['model'][k]) for k,v in model.state_dict().items())
            if arm!='standard':
                if initial_new is None:initial_new=initial['model']
                else:assert all(torch.equal(v,initial['model'][k]) for k,v in initial_new.items())
            for b in range(32):
                pre=load(ROOT/f'{arm}/batch_{b:03d}.pt');model.restore(pre)
                batch=load(ROOT/f'{arm}/data_{b:03d}.pt');entry=batch['update'];assert batch['arm']==arm and batch['batch']==b
                assert entry==report['history'][arm][b] and entry['batch']==b
                assert [e['key'] for e in batch['episodes']]==entry['keys']==list(TRAIN_KEYS[b*BATCH:(b+1)*BATCH])
                targets={r:[] for r in ('pricing','ordering')};seen=set();branch_total=0
                for item in batch['episodes']:
                    key=item['key'];calendar(env,key);budget.context=dict(arm=arm,batch=b,key=key)
                    reference=item['reference_rewards'] if arm=='standard' else item['whole_reference_rewards']
                    assert reference==references[key]
                    checked=(replay_training(model,'whole',env,item,budget.advance,seen) if arm=='whole' else replay_integrated(model,env,item,budget.advance,seen))
                    for n in totals:totals[n]+=checked[n]
                    for role in targets:targets[role]+=checked['targets'][role]
                    branch_total+=checked['branch_days']
                    context=dict(arm=arm,batch=b,key=key,root_role=None,root_day=None)
                    verify_main('actual',context,28,main_ledger)
                    for branch in checked['branch_plan']:
                        verify_main('branch',dict(context,root_role=branch['role'],root_day=branch['day']),branch['days'],main_ledger)
                assert branch_total==entry['branch_days']
                assert np.isclose(entry['stochastic_reward_day'],np.asarray([e['rewards'] for e in batch['episodes']]).mean(),rtol=1e-12)
                after=load(ROOT/(f'{arm}/batch_{b+1:03d}.pt' if b<31 else f'{arm}/final.pt'))
                assert torch.equal(after['rng'],torch.get_rng_state())
                assert set(entry['critic_targets'])==set(targets)
                for role in targets:
                    assert np.allclose(entry['critic_targets'][role],targets[role],rtol=5e-5,atol=3e-6)
                update_transition(model,arm,pre,after,entry,entry['critic_targets'])
                for role in ('pricing','ordering'):
                    rows=[r for ep in batch['episodes'] for r in ep['rows'][role]]
                    features=torch.tensor(np.asarray([r['features'] for r in rows]),dtype=torch.float32)
                    actions=torch.tensor(np.asarray([r['action'] for r in rows]),dtype=torch.long)
                    old=torch.tensor([r['logp'] if arm=='standard' else r['joint_logp'] for r in rows],dtype=torch.float32)
                    with torch.no_grad():
                        new=getattr(model,role).distribution(features).log_prob(actions).sum(-1)
                        delta=new-old;kl=float((delta.exp()-1-delta).mean())
                    assert np.isclose(kl,entry['metrics'][role+'_kl'],rtol=5e-5,atol=3e-6)
                atomic_json(OUT/'progress.json',dict(stage=stage,arm=arm,batches=b+1,observed_days=budget.completed,
                    reserved_days=budget.reserved,seconds=time.perf_counter()-began))
        assert next(main_ledger,None) is None and counts==report['parameter_counts']
        assert freeze['completed_days']==28672+57344+totals['branch_days']
        records={};stage='assessment'
        for arm in ('anchor',)+ARMS:
            model=policy(arm,env,covers);model.restore(load(ROOT/('baseline.pt' if arm=='anchor' else f'{arm}/final.pt')))
            saved=load(ROOT/f'assessment_{arm}.pt');assert [e['key'] for e in saved]==list(ASSESS_KEYS)
            records[arm]=[]
            for key,item in zip(ASSESS_KEYS,saved):
                calendar(env,key);budget.context=dict(arm=arm,batch=None,key=key)
                ep=replay_greedy(model,env,key,budget.advance,'assessment')
                assert ep['rewards']==item['rewards'] and serial(ep['trace'])==serial(item['trace'])
                records[arm].append(dict(key=key,**measure(ep,covers)))
                verify_main('assessment',dict(arm=arm,batch=None,key=key,root_role=None,root_day=None),28,assessment_ledger)
            for actual,stored in zip(records[arm],assessment['records'][arm]):equal(actual,stored)
        assert next(assessment_ledger,None) is None and main_steps==report['days']
        assert report['counts']==dict(reference=28672,actual=57344,branch=totals['branch_days'],assessment=8064)
        assert budget.pending is None and budget.completed==budget.reserved<=AUDIT_DAYS
        assert budget.counts['identity_check']<=3584 and budget.completed==report['days']+budget.counts['identity_check']
        result=independent_analysis(records);equal(result,report['analysis'])
        assert all(digest(ROOT/p)==h for p,h in allfiles.items()) and all(digest(p)==h for p,h in manifest['hashes'].items())
        if time.perf_counter()>deadline:raise TimeoutError('audit finalization time bound')
        budget.close()
        artifacts={p:digest(ROOT/p) for p in ('manifest.json','training_freeze.json','reference_freeze.json','assessment_start.json',
            'assessment_freeze.json','report.json','training_ledger.jsonl','assessment_ledger.jsonl')}
        atomic_json(OUT/'report.json',dict(status='PASS',verification_days=budget.completed,reserved_days=budget.reserved,
            counts=budget.counts,seconds=time.perf_counter()-began,training_episodes=2048,reference_episodes=1024,
            assessment_episodes=288,updates_checked=64,checkpoints_checked=67,main_steps_checked=main_steps,
            identity_reuse_disclosed=True,sgd_independently_refitted=False,analysis=result,totals=totals,
            artifacts=artifacts,verifier=digest(__file__),full_goal_achieved=False))
    except BaseException as exc:
        atomic_json(OUT/'failure.json',dict(status='failed',stage=stage,error=repr(exc),observed_days=budget.completed,
            reserved_days=budget.reserved,pending=budget.pending,seconds=time.perf_counter()-began));raise
    finally:
        if not budget.log.closed:budget.close()


if __name__=='__main__':main()
