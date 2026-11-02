"""Uninstalled fit artifact engine; no environment creation or fitting on import."""
from contextlib import contextmanager
import json
from pathlib import Path
import time
import numpy as np
import torch
from outputs.temperature_pilot_support import policy,temperature,training_collect,critic_targets,calendar,load
from outputs.prototype_temperature_local import TemperatureLocalPrototype
from outputs.temperature_pilot_replay import temperature_transition,ledger_pair
from outputs.integrated_control_replay import replay_integrated_control
from outputs.verify_team_credit_pilot import update_transition
from src.experiments.execution_journal import atomic_json,digest


@contextmanager
def information_guard(env,information):
    if information not in ('mappo','local'):raise ValueError('declared information required')
    original=env.get_state
    if information=='local':
        def forbidden():raise AssertionError('local construction/training/audit accessed global state')
        env.get_state=forbidden
    try:yield
    finally:env.get_state=original


def make_model(information,arm,env,covers,seed):
    if arm not in ('fixed','annealed'):raise ValueError('declared family required')
    with information_guard(env,information):
        return (policy(arm,env,covers,seed) if information=='mappo' else
            TemperatureLocalPrototype(env,covers,seed,arm=arm))


def contract(keys_by_batch,synthetic_fixture):
    keys=tuple(tuple(k) for k in keys_by_batch);flat=[k for row in keys for k in row]
    if len(flat)!=len(set(flat)):raise ValueError('unique actual episode keys across batches required')
    if synthetic_fixture:
        if len(keys)!=2 or any(len(k)!=1 for k in keys):raise ValueError('two-episode synthetic fixture only')
        return keys,(0,31),32
    if len(keys)!=32 or any(len(k)!=32 for k in keys):raise ValueError('32x32 unique actual episodes required')
    return keys,tuple(range(32)),1


def run_fit(root,model,env,references,keys_by_batch,torch_seed,budget,*,synthetic_fixture=False):
    keys,positions,repeats=contract(keys_by_batch,synthetic_fixture)
    if not {k for row in keys for k in row}<=set(references):raise ValueError('all same-key strong references required')
    root=Path(root);root.mkdir(parents=True,exist_ok=False);arm=model.temperature_arm
    info='local' if model.mode=='local' else 'mappo';began=budget.completed;started=time.perf_counter()
    checkpoints={};data={};history=[];torch.manual_seed(torch_seed)
    with information_guard(env,info):
        for b,(batch_keys,position) in enumerate(zip(keys,positions)):
            model.ordering.set_temperature(temperature(arm,position))
            name=f'batch_{b:03d}.pt';torch.save(model.checkpoint(),root/name);checkpoints[name]=digest(root/name)
            episodes=[]
            for key in batch_keys:
                if hasattr(env,'sim'):calendar(env,key)
                budget.context=dict(arm=info+'_'+arm,batch=b,key=key,root_role=None,root_day=None)
                ep=training_collect(model,arm,env,key,references[key],budget);ep['key']=key;episodes.append(ep)
            before={k:v.clone() for k,v in model.state_dict().items()};update_episodes=episodes*repeats
            metrics=model.update(update_episodes)
            name=f'after_{b:03d}.pt';torch.save(model.checkpoint(),root/name);checkpoints[name]=digest(root/name)
            changes={role:any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items() if k.startswith(role+'.'))
                for role in ('pricing','ordering','critics')}
            entry=dict(batch=b,position=position,keys=list(batch_keys),temperature=float(model.ordering.temperature),
                metrics=metrics,changes=changes,critic_targets=critic_targets(arm,update_episodes),
                branch_days=sum(e['reference_days'] for e in episodes))
            name=f'data_{b:03d}.pt';torch.save(dict(episodes=episodes,update=entry),root/name);data[name]=digest(root/name)
            history.append(entry)
        torch.save(model.checkpoint(),root/'final.pt');checkpoints['final.pt']=digest(root/'final.pt')
    assert budget.pending is None
    result=dict(status='complete',information=info,arm=arm,keys=[list(k) for k in keys],torch_seed=torch_seed,
        synthetic_fixture=synthetic_fixture,update_tensor_replication=repeats,updates=len(keys),
        unique_actual_episodes=sum(map(len,keys)),completed_days=budget.completed-began,
        reserved_days=budget.completed-began,checkpoints=checkpoints,data=data,history=history,
        seconds=time.perf_counter()-started)
    atomic_json(root/'fit_receipt.json',result);return result


def audit_fit(root,model,env,references,budget,stream,ledger_step,*,clone,stamp):
    root=Path(root);receipt=json.loads((root/'fit_receipt.json').read_text());arm=model.temperature_arm
    info='local' if model.mode=='local' else 'mappo'
    assert receipt['information']==info and receipt['arm']==arm and receipt['status']=='complete'
    keys,positions,repeats=contract(receipt['keys'],receipt['synthetic_fixture'])
    assert receipt['update_tensor_replication']==repeats and receipt['updates']==len(keys)
    expected_cp={'final.pt'}|{f'{prefix}_{b:03d}.pt' for prefix in ('batch','after') for b in range(len(keys))}
    assert set(receipt['checkpoints'])==expected_cp and set(receipt['data'])=={f'data_{b:03d}.pt' for b in range(len(keys))}
    allfiles={**receipt['checkpoints'],**receipt['data']}
    assert all(digest(root/p)==h for p,h in allfiles.items())
    torch.manual_seed(receipt['torch_seed']);initial=load(root/'batch_000.pt')
    assert torch.equal(initial['rng'],torch.get_rng_state())
    assert all(torch.equal(v,initial['model'][k]) for k,v in model.state_dict().items())
    totals=dict(branch_days=0,skips=0,selfchecks=0,common_pairs=0,same_price_pairs=0);main_days=0
    with information_guard(env,info):
        for b,(batch_keys,position) in enumerate(zip(keys,positions)):
            pre=load(root/f'batch_{b:03d}.pt');model.restore(pre)
            assert pre['ordering_temperature']==temperature(arm,position)
            saved=load(root/f'data_{b:03d}.pt');entry=saved['update'];episodes=saved['episodes']
            assert entry==receipt['history'][b] and entry['keys']==list(batch_keys) and entry['position']==position
            assert [e['key'] for e in episodes]==list(batch_keys)
            targets={r:[] for r in ('pricing','ordering')};seen=set();batch_days=0
            for key,ep in zip(batch_keys,episodes):
                if hasattr(env,'sim'):calendar(env,key)
                assert ep['whole_reference_rewards']==references[key]
                context=dict(arm=info+'_'+arm,batch=b,key=key,root_role=None,root_day=None);budget.context=dict(context)
                checked=replay_integrated_control(model,env,ep,budget.advance,seen,clone=clone,stamp=stamp)
                for n in totals:totals[n]+=checked[n]
                for r in targets:targets[r]+=checked['targets'][r]
                batch_days+=checked['branch_days']
                for _ in range(28):ledger_step=ledger_pair(stream,ledger_step,'actual',context);main_days+=1
                for branch in checked['branch_plan']:
                    for _ in range(branch['days']):
                        ledger_step=ledger_pair(stream,ledger_step,'branch',dict(context,root_role=branch['role'],root_day=branch['day']));main_days+=1
            assert entry['branch_days']==batch_days
            for r in targets:assert np.allclose(entry['critic_targets'][r],targets[r]*repeats,rtol=5e-5,atol=3e-6)
            after=load(root/f'after_{b:03d}.pt')
            assert torch.equal(after['rng'],torch.get_rng_state()) and after['ordering_temperature']==pre['ordering_temperature']==entry['temperature']
            update_transition(model,arm,pre,after,entry,entry['critic_targets'])
            for role in ('pricing','ordering'):
                rows=[r for e in episodes for r in e['rows'][role]]
                features=torch.tensor(np.asarray([r['features'] for r in rows]),dtype=torch.float32)
                actions=torch.tensor(np.asarray([r['action'] for r in rows]),dtype=torch.long)
                old=torch.tensor([r['joint_logp'] for r in rows],dtype=torch.float32)
                with torch.no_grad():
                    delta=getattr(model,role).distribution(features).log_prob(actions).sum(-1)-old
                    observed=float((delta.exp()-1-delta).mean())
                assert np.isclose(observed,entry['metrics'][role+'_kl'],rtol=5e-5,atol=3e-6)
            following=load(root/(f'batch_{b+1:03d}.pt' if b+1<len(keys) else 'final.pt'))
            next_tau=temperature(arm,positions[b+1]) if b+1<len(keys) else temperature(arm,position)
            temperature_transition(after,following,next_tau)
        model.restore(load(root/'final.pt'))
    assert main_days==receipt['completed_days']==receipt['reserved_days']
    assert all(digest(root/p)==h for p,h in allfiles.items())
    return dict(status='PASS',main_days=main_days,ledger_step=ledger_step,totals=totals,
        updates_checked=len(keys),checkpoints_checked=len(expected_cp),sgd_refitted=False)
