"""Archive-only independent reconstruction of the mixed-cadence pilot."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.train_compact_residual import digest,write_json
from scripts.verify_parameter_confirmation import episodes,check_episode
from src.config.sku_config import load_skus


def verify(output):
    out=Path(output);load=lambda p:json.loads(p.read_text())
    manifest,summary=load(out/'manifest.json'),load(out/'summary.json')
    for path,h in manifest['hashes'].items():assert digest(path)==h
    for name,h in summary['trace_hashes'].items():assert digest(out/name)==h
    calibration=next(p for p in manifest['hashes'] if p.endswith('train_only_nb_identifiable_v2.json'))
    order=load(Path(calibration))['series_order'];lookup={s.series_id:s.base_price for s in load_skus()}
    base=np.array([lookup[s] for s in order]);multipliers=np.array([.8,.9,1.,1.1,1.2])
    initial=np.r_[manifest['initial_covers'],np.full(10,4.),np.zeros(11)]
    sigma=np.r_[np.full(20,.75),np.full(11,.5)];parameters={};scores={};finals={}
    for arm,updates,first in [('joint',24,1700000),('ordering',12,1701000),('pricing',12,1702000)]:
        theta=initial.copy();m=np.zeros(31);v=np.zeros(31);active=np.ones(31,bool)
        if arm=='ordering':active[10:]=False
        if arm=='pricing':active[:10]=False
        rng=np.random.default_rng(501);history=load(out/f'{arm}_history.json')
        assert len(history)==updates
        for u,record in enumerate(history):
            e=rng.normal(size=(8,31))*active
            np.testing.assert_array_equal(e,record['epsilon']);np.testing.assert_array_equal(theta,record['mean_before'])
            differences=[]
            for pair,r in enumerate(record['records']):
                key=first+u*8+pair;assert key==r['key']
                for sign,direction in [('plus',1),('minus',-1)]:
                    label=f'{arm}_u{u}_p{pair}_{sign}';parameters[label,key]=theta+direction*sigma*e[pair];scores[label,key]=r[sign]
                differences.append(r['plus']['discounted_day']-r['minus']['discounted_day'])
            np.testing.assert_array_equal(differences,record['differences'])
            gradient=np.mean(np.asarray(differences)[:,None]*e/(2*sigma),axis=0)*active
            np.testing.assert_array_equal(gradient,record['gradient'])
            m=.9*m+.1*gradient;v=.999*v+.001*gradient**2
            theta+=.1*(m/(1-.9**(u+1)))/(np.sqrt(v/(1-.999**(u+1)))+1e-8)
            theta[:10]=theta[:10].clip(0,16);theta[10:20]=theta[10:20].clip(0,4);theta[20:]=theta[20:].clip(-4,4)
            np.testing.assert_array_equal(theta,record['state']['mean'])
        finals[arm]=theta.copy();np.testing.assert_array_equal(theta,load(out/f'{arm}_final.json')['mean'])
        assert digest(out/f'{arm}_final.json')==load(out/'frozen_checkpoints.json')[arm]
        if arm=='ordering':np.testing.assert_array_equal(theta[10:],initial[10:])
        if arm=='pricing':np.testing.assert_array_equal(theta[:10],initial[:10])
    def audit(trace,p):
        p=np.asarray(p,np.float32);key=trace[0]['scenario']
        start=int(np.random.default_rng(np.random.SeedSequence([key,314159])).integers(15,1859))
        result=check_episode(trace,p[:10],start);prices=[];counts=np.zeros(5,int)
        for r in trace:
            if r['day'] in [1,8,15,22]:
                obs=np.asarray(r['pricing_obs'],np.float32)
                z=np.minimum(np.maximum(obs[:10]/(1+7*obs[-20:-10]),0),2)-1
                score=p[10:20]+p[20:30]*z+p[30]*np.float32((29-r['day'])/28-.5)
                action=np.floor(score.clip(0,4)+.5).astype(int)
                np.testing.assert_array_equal(action,r['price_actions']);prices=base*multipliers[action]
                counts+=np.bincount(action,minlength=5)
            else:assert r['pricing_obs'] is None and r['price_actions'] is None
            np.testing.assert_allclose(prices,r['info']['price'],atol=1e-12,rtol=0)
        result['price_counts']=counts.tolist()
        return result
    count=0;pair_noise=None
    for key,trace in episodes(out/'training_traces.jsonl'):
        result=audit(trace,parameters[key]);expected=scores[key]
        for metric in ['reward_day','discounted_day','fill']:assert abs(result[metric]-expected[metric])<1e-10
        assert result['price_counts']==expected['price_counts']
        noise=[(r['demand_seed'],r['supplier_seed']) for r in trace]
        if key[0].endswith('_plus'):pair_noise=(key[1],noise)
        else:assert pair_noise==(key[1],noise)
        count+=1
    assert count==768
    combined=finals['ordering'].copy();combined[10:]=finals['pricing'][10:]
    policies=dict(finals,independent=combined,start=initial,full7_5=np.r_[np.full(10,7.5),initial[10:]])
    values={};noise_by_key={};replay_targets={}
    for key,trace in episodes(out/'assessment_traces.jsonl'):
        assert key not in values;values[key]=audit(trace,policies[key[0]])
        noise=[(r['demand_seed'],r['supplier_seed']) for r in trace]
        if key[1] in noise_by_key:assert noise_by_key[key[1]]==noise
        else:noise_by_key[key[1]]=noise
        if key[1]==1710000 and key[0] in ['joint','start']:replay_targets[key]=trace
    assert len(values)==384 and sorted(noise_by_key)==list(range(1710000,1710064))
    for row in summary['rows']:
        value=values[row['label'],row['scenario']]
        for metric in ['reward_day','discounted_day','fill']:assert abs(value[metric]-row[metric])<1e-10
        assert value['price_counts']==row['price_counts']
    replays=0
    for key,trace in episodes(out/'replays.jsonl'):
        assert trace==replay_targets[key];replays+=1
    assert replays==2
    rng=np.random.default_rng(20261022);indices=rng.integers(0,64,size=(10000,64));eligible=True
    for ref in ['start','independent']:
        for metric in ['reward_day','fill']:
            x=np.array([values['joint',k][metric]-values[ref,k][metric] for k in range(1710000,1710064)])
            result=summary['analysis']['effects'][f'{ref}_{metric}']
            assert abs(result['mean']-x.mean())<1e-10
            bounds=np.quantile([x[i].mean() for i in indices],[.00625,.99375])
            np.testing.assert_allclose(bounds,result['ci9875'],atol=1e-10,rtol=0)
            eligible &= bool(x.mean()>=.5 if metric=='reward_day' else x.mean()>=-.005)
    assert eligible==summary['analysis']['eligible_for_new_confirmation']
    for label in policies:
        group=[values[label,k] for k in range(1710000,1710064)]
        for metric in ['reward_day','fill']:
            assert abs(np.mean([r[metric] for r in group])-summary['analysis']['means'][label][metric])<1e-10
        assert np.sum([r['price_counts'] for r in group],axis=0).tolist()==summary['analysis']['price_counts'][label]
    assert summary['counters']==dict(episodes=1154,days=32312) and summary['seconds']<=900 and not summary['test_scoring']
    write_json(out/'verification.json',dict(status='passed',updates=48,training_episodes=768,assessment_episodes=384,
        exact_replays=2,independent_actions_gradients_returns_intervals=True,paired_exogenous_seeds=True,
        eligible_for_new_confirmation=eligible,new_simulator_days=0,verifier_sha256=digest(__file__)))
    print('Joint pilot archive verification passed.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);verify(p.parse_args().output)
