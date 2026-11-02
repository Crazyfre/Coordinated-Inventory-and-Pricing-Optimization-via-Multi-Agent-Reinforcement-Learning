"""Independent latent-plan likelihood/PPO reconstruction and rollout audit."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scripts.verify_structured_ordering_ppo import equal_state
from scripts.verify_parameter_confirmation import episodes,check_episode
from scripts.train_compact_residual import digest,write_json
from src.config.sku_config import load_skus


def independent_update(mean,optimizer,plans,returns):
    z=torch.tensor(plans,dtype=torch.float32);r=np.asarray(returns,float)
    baseline=np.array([np.mean(np.delete(r.mean(1),i)) for i in range(8)])
    adv=(r-baseline[:,None]).reshape(-1);adv=((adv-adv.mean())/(adv.std()+1e-8)).astype(np.float32)
    advantage=torch.tensor(adv)
    def logp():return (-.5*((z-mean)/.75)**2-np.log(.75*np.sqrt(2*np.pi))).sum((1,2))
    old=logp().detach().clone();steps=0
    for _ in range(4):
        lp=logp();ratio=torch.exp(lp-old)
        if float(((ratio-1)-(lp-old)).mean().detach())>.02:break
        loss=-torch.minimum(ratio*advantage,torch.clamp(ratio,.8,1.2)*advantage).mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([mean],.5);optimizer.step();steps+=1
    return old.tolist(),adv.tolist(),steps


def verify(output):
    out=Path(output);load=lambda p:json.loads((out/p).read_text());read=lambda p:torch.load(out/p,weights_only=True,map_location='cpu')
    manifest,summary=load('manifest.json'),load('summary.json');assert summary['status']=='complete';torch.set_num_threads(1)
    for p,h in manifest['hashes'].items():assert digest(p)==h,p
    for field in ['archive_hashes','trace_hashes']:
        for p,h in summary[field].items():assert digest(out/p)==h
    for p,h in load('assessment_freeze.json').items():assert digest(out/p)==h
    covers=np.asarray(manifest['covers'],np.float32);noise={}
    calibration=next(p for p in manifest['hashes'] if p.endswith('train_only_nb_identifiable_v2.json'))
    order=json.loads(Path(calibration).read_text())['series_order'];lookup={s.series_id:s.base_price for s in load_skus()}
    prices=np.array([lookup[s] for s in order])*1.2

    def audit(trace,plan):
        key=trace[0]['scenario'];start=int(np.random.default_rng(np.random.SeedSequence([key,314159])).integers(15,1859))
        score=check_episode(trace,start=start)
        for i,r in enumerate(trace):
            c=np.asarray(plan[0 if len(plan)==1 else i],np.float32)
            np.testing.assert_array_equal(c,r['latent']);x=np.array(r['obs'],np.float32)
            a=np.ceil(np.maximum(c.clip(0,20)*x[-10:]-(x[:10]+x[10:80].reshape(10,7).sum(1)),0)/5).clip(0,40).astype(int)
            np.testing.assert_array_equal(a,r['actions']);np.testing.assert_allclose(prices,r['info']['price'],atol=1e-12,rtol=0)
        current=[(r['demand_seed'],r['supplier_seed']) for r in trace]
        if key in noise:assert noise[key]==current
        else:noise[key]=current
        score['discounted_day']=sum(.99**i*r['reward'] for i,r in enumerate(trace))/28
        return score

    stream=iter(episodes(out/'training.jsonl'));finals={}
    for arm,T in [('episode',1),('daily',28)]:
        mean=torch.nn.Parameter(torch.tensor(covers.copy()));optimizer=torch.optim.Adam([mean],lr=.03)
        rng=np.random.default_rng(801);initial=read(f'{arm}_initial.pt')
        equal_state(dict(mean=mean.detach(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,arm=arm),initial)
        for u in range(64):
            record=read(f'{arm}_u{u:03d}.pt');e=rng.normal(size=(8,T,10))
            np.testing.assert_array_equal(e,record['epsilon']);center=mean.detach().numpy().astype(float)
            plans=np.stack([center+.75*e,center-.75*e],1).reshape(16,T,10).astype(np.float32)
            np.testing.assert_array_equal(plans,record['plans']);returns=[]
            for i,plan in enumerate(plans):
                key,trace=next(stream);assert key==(f'{arm}_u{u}_p{i}',2340000+u*8+i//2)
                returns.append(audit(trace,plan)['discounted_day'])
            returns=np.array(returns).reshape(8,2);np.testing.assert_array_equal(returns,record['returns'])
            old,adv,steps=independent_update(mean,optimizer,plans,returns)
            assert old==record['metrics']['old_logp'] and adv==record['metrics']['advantage'] and steps==record['metrics']['steps']
            equal_state(dict(mean=mean.detach(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,arm=arm),record['state'])
        finals[arm]=mean.detach().numpy().copy()
    assert next(stream,None) is None
    results={};targets={}
    for key,trace in episodes(out/'assessment.jsonl'):
        label,k=key;assert key not in results and k in range(2350000,2350064)
        if label=='baseline':plan=covers[None]
        else:
            arm,mode=label.split('_');assert arm in finals and mode in ['initial','greedy','sampled']
            T=1 if arm=='episode' else 28;center=covers if mode=='initial' else finals[arm]
            if mode=='greedy':plan=np.broadcast_to(center,(T,10)).copy()
            else:
                rng=np.random.default_rng(np.random.SeedSequence([801,k,900]))
                plan=(center.astype(float)+.75*rng.normal(size=(T,10))).astype(np.float32)
        results[key]=audit(trace,plan)
        if k==2350000 and label.endswith(('greedy','sampled')):targets[key]=trace
    assert len(results)==448
    seen=set()
    for key,trace in episodes(out/'replays.jsonl'):
        assert key not in seen and trace==targets[key];seen.add(key)
    assert len(seen)==4
    assert len(summary['rows'])==len({(r['label'],r['scenario']) for r in summary['rows']})==448
    for r in summary['rows']:
        for metric,value in results[r['label'],r['scenario']].items():assert abs(value-r[metric])<1e-10
    draws=np.random.default_rng(20261030).integers(64,size=(10000,64));analysis=summary['analysis']
    pairs=[(f'{a}_{m}',ref) for a in ['episode','daily'] for m in ['greedy','sampled'] for ref in ['baseline',f'{a}_initial']]
    pairs += [('episode_'+m,'daily_'+m) for m in ['greedy','sampled']]
    assert set(analysis['effects'])=={f'{a}-{b}:{m}' for a,b in pairs for m in ['reward_day','fill']}
    for a,b in pairs:
        for m in ['reward_day','fill']:
            d=np.array([results[a,k][m]-results[b,k][m] for k in range(2350000,2350064)])
            expected=analysis['effects'][f'{a}-{b}:{m}'];assert abs(d.mean()-expected['mean'])<1e-10
            np.testing.assert_allclose(np.quantile([d[i].mean() for i in draws],[.025,.975]),expected['ci95'],atol=1e-10,rtol=0)
    for label,values in analysis['means'].items():
        for m,value in values.items():assert abs(value-np.mean([results[label,k][m] for k in range(2350000,2350064)]))<1e-10
    eligible=[f'{a}_{m}' for a in ['episode','daily'] for m in ['greedy','sampled']
        if analysis['effects'][f'{a}_{m}-baseline:reward_day']['mean']>=.5
        and analysis['effects'][f'{a}_{m}-baseline:fill']['mean']>=-.005
        and analysis['effects'][f'{a}_{m}-{a}_initial:reward_day']['mean']>0]
    assert eligible==analysis['promising']
    assert summary['counters']==dict(episodes=2500,days=70000) and summary['seconds']<=1800 and not summary['test_scoring']
    write_json(out/'verification.json',dict(status='passed',independent_ppo_updates=128,training_episodes=2048,
        assessment_episodes=448,exact_replays=4,plans_rng_likelihood_optimizer_actions_returns_pairing_intervals=True,
        new_simulator_days=0,verifier_sha256=digest(__file__)))
    print('Temporal cover PPO comparison verified.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);verify(p.parse_args().output)
