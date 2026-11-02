"""Independent fresh-seed temporal PPO training and crossed-assessment verifier."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scripts.verify_temporal_cover_ppo import independent_update
from scripts.verify_structured_ordering_ppo import equal_state
from scripts.verify_parameter_confirmation import episodes,check_episode
from scripts.train_compact_residual import digest,write_json
from src.config.sku_config import load_skus

SEEDS=(811,822,833,844,855)
KEYS=tuple(range(2420000,2420128))


def verify_analysis(index,analysis):
    expected_labels=['baseline']+[f'{a}_{s}' for a in ['episode','daily'] for s in SEEDS]
    assert set(index)=={(l,k) for l in expected_labels for k in KEYS}
    rng=np.random.default_rng(20261031)
    rs=rng.integers(0,5,size=(10000,5));cs=rng.integers(0,128,size=(10000,128))
    assert set(analysis['effects'])=={f'{r}:{m}' for r in ['baseline','daily'] for m in ['reward_day','fill']}
    computed={}
    for ref in ['baseline','daily']:
        for metric in ['reward_day','fill']:
            x=np.array([[index[f'episode_{s}',k][metric]-index['baseline' if ref=='baseline' else f'daily_{s}',k][metric]
                         for k in KEYS] for s in SEEDS])
            draws=np.array([x[r][:,c].mean() for r,c in zip(rs,cs)])
            bounds=np.quantile(draws,[.00625,.99375]);e=analysis['effects'][f'{ref}:{metric}']
            assert abs(e['mean']-x.mean())<1e-10
            np.testing.assert_allclose(e['seed_means'],x.mean(1),atol=1e-10,rtol=0)
            np.testing.assert_allclose(e['ci9875'],bounds,atol=1e-10,rtol=0)
            computed[ref,metric]=(float(x.mean()),bounds,x.mean(1))
    r,f=computed['baseline','reward_day'],computed['baseline','fill']
    primary=r[0]>=.5 and r[1][0]>0 and f[1][0]>=-.005 and min(r[2])>0
    r,f=computed['daily','reward_day'],computed['daily','fill']
    matched=r[1][0]>0 and f[1][0]>=-.005
    assert analysis['primary_pass']==bool(primary) and analysis['matched_superiority_pass']==bool(matched)
    assert set(analysis['means'])==set(expected_labels)
    for label,metrics in analysis['means'].items():
        assert set(metrics)=={'reward_day','fill'}
        for m,v in metrics.items():assert abs(v-np.mean([index[label,k][m] for k in KEYS]))<1e-10


def verify(output):
    out=Path(output);load=lambda p:json.loads((out/p).read_text())
    read=lambda p:torch.load(out/p,weights_only=True,map_location='cpu')
    manifest,summary,freeze=load('manifest.json'),load('summary.json'),load('assessment_freeze.json')
    assert summary['status']=='complete' and manifest['seeds']==list(SEEDS);torch.set_num_threads(1)
    for p,h in manifest['hashes'].items():assert digest(p)==h,p
    assert 'src/config/skus.json' in manifest['hashes']
    assert any(p.endswith('aligned_skus.csv') for p in manifest['hashes'])
    for field in ['archive_hashes','trace_hashes']:
        for p,h in summary[field].items():assert digest(out/p)==h
    expected_finals={f'{a}_{s}_u063.pt' for a in ['episode','daily'] for s in SEEDS}
    assert set(freeze['checkpoints'])==expected_finals and freeze['source_hashes']==manifest['hashes']
    assert freeze['counters']==dict(episodes=10240,days=286720)
    for p,h in freeze['checkpoints'].items():assert digest(out/p)==h
    covers=np.asarray(manifest['covers'],np.float32)
    calibration=next(p for p in manifest['hashes'] if p.endswith('train_only_nb_identifiable_v2.json'))
    order=json.loads(Path(calibration).read_text())['series_order'];lookup={s.series_id:s.base_price for s in load_skus()}
    prices=np.array([lookup[s] for s in order])*1.2;noise={}

    def audit(trace,plan):
        key=trace[0]['scenario'];start=int(np.random.default_rng(np.random.SeedSequence([key,314159])).integers(15,1859))
        score=check_episode(trace,start=start)
        for i,r in enumerate(trace):
            c=np.asarray(plan[0 if len(plan)==1 else i],np.float32);np.testing.assert_array_equal(c,r['latent'])
            x=np.asarray(r['obs'],np.float32)
            a=np.ceil(np.maximum(c.clip(0,20)*x[-10:]-(x[:10]+x[10:80].reshape(10,7).sum(1)),0)/5).clip(0,40).astype(int)
            np.testing.assert_array_equal(a,r['actions']);np.testing.assert_allclose(prices,r['info']['price'],atol=1e-12,rtol=0)
        current=[(r['demand_seed'],r['supplier_seed']) for r in trace]
        if key in noise:assert noise[key]==current
        else:noise[key]=current
        score['discounted_day']=sum(.99**i*r['reward'] for i,r in enumerate(trace))/28
        return score

    stream=iter(episodes(out/'training.jsonl'));finals={};updates=0
    for j,seed in enumerate(SEEDS):
        for arm,T in [('episode',1),('daily',28)]:
            label=f'{arm}_{seed}';mean=torch.nn.Parameter(torch.tensor(covers.copy()))
            optimizer=torch.optim.Adam([mean],lr=.03);rng=np.random.default_rng(seed)
            equal_state(dict(mean=mean.detach(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,arm=arm),read(f'{label}_initial.pt'))
            for u in range(64):
                record=read(f'{label}_u{u:03d}.pt');e=rng.normal(size=(8,T,10))
                np.testing.assert_array_equal(e,record['epsilon']);center=mean.detach().numpy().astype(float)
                plans=np.stack([center+.75*e,center-.75*e],1).reshape(16,T,10).astype(np.float32)
                np.testing.assert_array_equal(plans,record['plans']);returns=[]
                for i,plan in enumerate(plans):
                    key,trace=next(stream);assert key==(f'{label}_u{u}_p{i}',2360000+j*10000+u*8+i//2)
                    returns.append(audit(trace,plan)['discounted_day'])
                returns=np.array(returns).reshape(8,2);np.testing.assert_array_equal(returns,record['returns'])
                old,adv,steps=independent_update(mean,optimizer,plans,returns)
                assert old==record['metrics']['old_logp'] and adv==record['metrics']['advantage'] and steps==record['metrics']['steps']
                equal_state(dict(mean=mean.detach(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,arm=arm),record['state']);updates+=1
            finals[label]=mean.detach().numpy().copy();print(f'verified {label}',flush=True)
    assert next(stream,None) is None and updates==640
    results={};targets={}
    for key,trace in episodes(out/'assessment.jsonl'):
        label,k=key;assert key not in results and k in KEYS and label in ['baseline',*finals]
        plan=(covers if label=='baseline' else finals[label])[None]
        results[key]=audit(trace,plan)
        if k==KEYS[0] and label!='baseline':targets[key]=trace
    assert len(results)==1408
    seen=set()
    for key,trace in episodes(out/'replays.jsonl'):
        assert key not in seen and trace==targets[key];seen.add(key)
    assert len(seen)==10
    assert len(summary['rows'])==len({(r['label'],r['scenario']) for r in summary['rows']})==1408
    for r in summary['rows']:
        for metric,value in results[r['label'],r['scenario']].items():assert abs(value-r[metric])<1e-10
    verify_analysis(results,summary['analysis'])
    assert summary['counters']==dict(episodes=11658,days=326424)
    assert set(summary['timings'])==set(finals) and all(0<t<=900 for t in summary['timings'].values())
    assert summary['training_seconds']==freeze['training_seconds']<=6000
    assert summary['assessment_seconds']<=1800 and summary['seconds']<=7800 and not summary['test_scoring']
    write_json(out/'verification.json',dict(status='passed',independent_ppo_updates=640,training_episodes=10240,
        assessment_episodes=1408,exact_replays=10,plans_rng_optimizer_actions_economics_pairing_crossed_intervals=True,
        new_simulator_days=0,primary_pass=summary['analysis']['primary_pass'],
        matched_superiority_pass=summary['analysis']['matched_superiority_pass'],verifier_sha256=digest(__file__)))
    print('Temporal confirmation verified.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);verify(p.parse_args().output)
