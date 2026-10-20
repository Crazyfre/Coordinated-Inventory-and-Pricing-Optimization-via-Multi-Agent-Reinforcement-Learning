"""Independent archive checks for the frozen review characterization."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.train_compact_residual import digest, write_json
from scripts.verify_parameter_confirmation import episodes, check_episode


def verify(output):
    out=Path(output);load=lambda p:json.loads(p.read_text())
    manifest,summary=load(out/'manifest.json'),load(out/'summary.json')
    for path,expected in manifest['hashes'].items(): assert digest(path)==expected
    for name,expected in summary['trace_hashes'].items(): assert digest(out/name)==expected
    study=Path(manifest['study']);seeds=[211,222,233,244,255]
    final={f'learned_{s}':np.array(load(study/f'sku_seed{s}_final.json')['mean']) for s in seeds}
    progress={'full7_5':np.full(10,7.5)}
    for s in seeds:
        history=load(study/f'sku_seed{s}_history.json')
        for u in [16,32,48,64]:progress[f'u{u}_{s}']=np.array(history[u-1]['state']['mean'])
        np.testing.assert_array_equal(progress[f'u64_{s}'],final[f'learned_{s}'])
    policies=dict(final,full7_5=np.full(10,7.5));count=0;replays=0;all_values={}
    for name in ['progress']+[c[0] for c in manifest['conditions']]:
        source=load(out/('learning_progress.json' if name=='progress' else f'{name}_metrics.json'))
        expected={(r['label'],r['scenario']):r for r in source['rows']}
        observed={};noise={};targets={}
        for key,trace in episodes(out/f'{name}_traces.jsonl'):
            assert key not in observed
            parameters=(progress if name=='progress' else policies).get(key[0])
            start=int(np.random.default_rng(np.random.SeedSequence([key[1],314159])).integers(15,1859)) if name=='progress' else 1886
            result=check_episode(trace,parameters,start)
            e=expected[key]
            for metric in ['reward_day','fill']: assert abs(result[metric]-e[metric])<1e-10
            assert abs(e['episodic_return']-28*result['reward_day'])<1e-8
            assert e['success']==int(result['fill']>=.95)
            assert e['start']==start
            for term in ['revenue','ordering_cost','holding_cost','stockout_cost','price_change_cost']:
                assert abs(sum(r['info']['reward_terms'][term] for r in trace)-e[term])<1e-8
            for metric,info in [('demand','demand'),('sales','sales'),('lost_sales','lost_sales'),
                                ('clipped_units','orders_clipped'),('delivered_units','orders_delivered')]:
                assert abs(sum(np.asarray(r['info'][info]).sum() for r in trace)-e[metric])<1e-8
            assert abs(np.mean([r['info']['total_on_hand'] for r in trace])-e['mean_inventory'])<1e-10
            assert abs(np.mean([np.asarray(r['info']['lost_sales'])>0 for r in trace])-e['stockout_sku_day_fraction'])<1e-10
            assert all(r['info']['total_on_hand'] <= (250 if name=='S8' else 500)+1e-8 for r in trace)
            demand=[(r['info']['demand'],r['info']['price']) for r in trace]
            if key[1] in noise: assert noise[key[1]]==demand
            else: noise[key[1]]=demand
            observed[key]=result;count+=1
            if name=='progress' and key==('u64_211',1600000):targets[key]=trace
            if name!='progress' and key[1]==1610000 and key[0] in ['learned_211','full7_5']:targets[key]=trace
        assert observed.keys()==expected.keys()
        keys=list(range(1600000,1600032)) if name=='progress' else list(range(1610000,1610032))
        assert sorted(noise)==keys and len(observed)==(672 if name=='progress' else 288)
        replay_path=out/('progress_replay.jsonl' if name=='progress' else f'{name}_replays.jsonl')
        seen=set()
        for key,trace in episodes(replay_path):
            assert key not in seen and trace==targets[key];seen.add(key);replays+=1
        assert seen==targets.keys()
        all_values[name]=observed
        if name!='progress':
            for metric in ['reward_day','fill']:
                values=np.array([[observed[f'learned_{s}',k][metric]-observed['full7_5',k][metric] for k in keys] for s in seeds])
                rng=np.random.default_rng(20261020)
                rs=rng.integers(5,size=(10000,5));cs=rng.integers(32,size=(10000,32))
                draws=[values[np.ix_(r,c)].mean() for r,c in zip(rs,cs)]
                endpoint=summary['analysis'][name]['effects_vs_full7_5'][metric]
                np.testing.assert_allclose(np.quantile(draws,[.025,.975]),endpoint['ci95'],atol=1e-10,rtol=0)
                np.testing.assert_allclose(values.mean(1),endpoint['seed_means'],atol=1e-10,rtol=0)
                assert abs(values.mean()-endpoint['mean'])<1e-10
            for label,metrics in summary['analysis'][name]['controllers'].items():
                x=np.array([28*observed[label,k]['reward_day'] for k in keys])
                ret=metrics['episodic_return']
                for field,value in [('mean',x.mean()),('median',np.median(x)),('std',x.std(ddof=1)),('p05',np.quantile(x,.05))]:
                    assert abs(ret[field]-value)<1e-8
                assert metrics['success']['successes']==sum(observed[label,k]['fill']>=.95 for k in keys)
        print(f'Verified {name} actions, metrics, pairing and replays.',flush=True)
    assert all_values['S2']==all_values['S6']==all_values['S4_e15']
    for seed in seeds:
        run=0;onset=None;confirmed=None
        for u in [0,16,32,48,64]:
            gain=0. if u==0 else np.mean([all_values['progress'][f'u{u}_{seed}',k]['reward_day']-
                all_values['progress']['full7_5',k]['reward_day'] for k in range(1600000,1600032)])
            stored=summary['progress'][str(seed)]['evaluations'][u//16]
            assert abs(stored['mean_evaluation_return']-gain)<1e-10
            assert stored['steps']==u*16*28
            run=run+1 if gain>=1. else 0
            if run>=3 and confirmed is None:onset=(u-32)*16*28;confirmed=u*16*28
        threshold=summary['progress'][str(seed)]['threshold']
        assert threshold['onset_step']==onset and threshold['confirmed_step']==confirmed
    rng=np.random.default_rng(20261021)
    for metric in ['reward_day','fill']:
        a=np.array([[all_values['progress'][f'u64_{s}',k][metric] for k in range(1600000,1600032)] for s in seeds])
        b=np.array([[all_values['S2'][f'learned_{s}',k][metric] for k in range(1610000,1610032)] for s in seeds])
        draws=[]
        for _ in range(10000):
            r=rng.integers(0,5,5);c=rng.integers(0,32,32);d=rng.integers(0,32,32)
            draws.append(a[np.ix_(r,c)].mean()-b[np.ix_(r,d)].mean())
        gap=summary['train_validation_gap'][metric]
        assert abs(a.mean()-b.mean()-gap['train_minus_validation'])<1e-10
        np.testing.assert_allclose(np.quantile(draws,[.025,.975]),gap['ci95'],atol=1e-10,rtol=0)
    assert count==3552 and replays==21 and summary['counters']==dict(episodes=3573,days=100044)
    assert summary['seconds']<=2400 and summary['training_updates']==0 and not summary['test_scoring']
    write_json(out/'verification.json',dict(status='passed',checked_episodes=count,exact_replays=replays,
        source_trace_hashes=True,primary_actions=True,reward_operations_pairing=True,robustness_intervals=True,
        progress_threshold=True,train_validation_gap=True,new_simulator_days=0,test_scoring=False,
        verifier_sha256=digest(__file__)))
    print('Characterization archive verification passed.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);verify(p.parse_args().output)
