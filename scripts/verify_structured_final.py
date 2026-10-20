"""Archive-only independent checks of the final held-out matrix."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t
from scripts.train_compact_residual import digest, write_json
from scripts.verify_parameter_confirmation import episodes, check_episode


def check_baseline(row,coefficients):
    """Reconstruct projected actions; bound heuristic rounding from float32 logs.

    Heuristics use float64 simulator inputs while observations are float32.
    Adjacent float32 values conservatively enclose the original inputs.
    """
    x=np.asarray(row['obs'],np.float32);label=row['label'];actual=np.asarray(row['actions'])
    if label=='projected':
        tx=torch.as_tensor(x)
        pos=(tx[:10]+tx[10:80].reshape(10,7).sum(-1)).numpy()
        features=np.stack([x[-10:].astype(float)/5,pos.astype(float)/5,np.ones(10)],axis=-1)
        center=np.ceil(np.maximum(features@coefficients,0)).clip(0,40)
        target=np.ceil(np.maximum(np.float32(7.5)*x[-10:]-pos,0)/np.float32(5)).clip(0,40)
        np.testing.assert_array_equal(actual,np.minimum(np.maximum(target,(center-2).clip(0,40)),(center+2).clip(0,40)))
        return 0
    if label not in ['constant','ss']:return 0
    lo=np.nextafter(x,np.float32(-np.inf)).astype(float)
    hi=np.nextafter(x,np.float32(np.inf)).astype(float)
    ml,mh=lo[-10:],hi[-10:]
    pl=lo[:10]+lo[10:80].reshape(10,7).sum(1)
    ph=hi[:10]+hi[10:80].reshape(10,7).sum(1)
    if label=='constant':ql,qh=ml,mh
    else:
        ql=np.where(ph<7*ml,np.maximum(8*ml-ph,0),0)
        qh=np.where(pl<7*mh,np.maximum(8*mh-pl,0),0)
    lower=np.rint((ql/5).clip(0,40));upper=np.rint((qh/5).clip(0,40))
    assert np.all((actual>=lower)&(actual<=upper))
    return int(np.count_nonzero(lower!=upper))


def check_effect(values, endpoint, coverage, seed):
    values=np.asarray(values,float)
    assert values.shape==(10,50)
    rng=np.random.default_rng(seed)
    rows=rng.integers(10,size=(10000,10));cols=rng.integers(50,size=(10000,50))
    draws=[values[np.ix_(r,c)].mean() for r,c in zip(rows,cols)]
    alpha=(1-coverage)/2
    np.testing.assert_allclose(np.quantile(draws,[alpha,1-alpha]),endpoint['interval'],atol=1e-10,rtol=0)
    np.testing.assert_allclose(values.mean(1),endpoint['seed_means'],atol=1e-10,rtol=0)
    assert abs(values.mean()-endpoint['mean'])<1e-10 and endpoint['coverage']==coverage


def verify(output):
    out=Path(output);load=lambda p:json.loads(p.read_text())
    manifest=load(out/'frozen_manifest.json');summary=load(out/'summary.json')
    assert digest(out/'frozen_manifest.json')==summary['frozen_manifest_sha256']
    assert manifest['status']=='frozen_before_test' and manifest['start']==1914
    for path,h in manifest['hashes'].items():assert digest(path)==h
    for name,h in summary['trace_hashes'].items():assert digest(out/name)==h
    training=Path(manifest['training'])
    assert digest(training/'summary.json')==manifest['training_summary_sha256']
    assert digest(training/'verification.json')==manifest['training_verification_sha256']
    seeds=[211,222,233,244,255,266,277,288,299,310];keys=list(range(2100000,2100050))
    labels=[f'{c}_{s}' for c in ['main','pooled','independent'] for s in seeds]
    assert set(manifest['policies'])==set(labels)
    policies={}
    for label,record in manifest['policies'].items():
        assert digest(record['path'])==record['sha256']
        state=load(Path(record['path']));assert state['steps']==64
        p=np.asarray(state['mean'])
        if label.startswith('pooled_'):p=np.repeat(p,10)
        np.testing.assert_array_equal(p,record['parameters']);policies[label]=p
    policies['full7_5']=np.full(10,7.5)
    labels+=['full7_5','projected','constant','ss']
    required={(label,key) for label in labels for key in keys}
    from scripts.train_compact_residual import INPUTS
    coefficients=np.array(load(Path(INPUTS['center']))['coefficients'])
    count=replays=ambiguous=0;all_values={}
    assert [c[0] for c in manifest['conditions']]==['S1','S2','S3','S4_e05','S4_e15','S4_e30','S5','S6','S7','S8']
    for name,scenario,elasticity in manifest['conditions']:
        saved=load(out/f'{name}_metrics.json');analysis=summary['analysis'][name]
        assert saved['analysis']==analysis and len(saved['rows'])==1700
        expected={(r['label'],r['scenario']):r for r in saved['rows']}
        assert expected.keys()==required
        observed={};noise={};targets={}
        for key,trace in episodes(out/f'{name}_traces.jsonl'):
            assert key not in observed and key in required
            result=check_episode(trace,policies.get(key[0]),1914);e=expected[key]
            for row in trace:ambiguous+=check_baseline(row,coefficients)
            for metric in ['reward_day','fill']:assert abs(result[metric]-e[metric])<1e-10
            assert e['start']==1914 and e['success']==int(result['fill']>=.95)
            assert abs(e['episodic_return']-28*result['reward_day'])<1e-8
            for term in ['revenue','ordering_cost','holding_cost','stockout_cost','price_change_cost']:
                assert abs(sum(r['info']['reward_terms'][term] for r in trace)-e[term])<1e-8
            for metric,field in [('demand','demand'),('sales','sales'),('lost_sales','lost_sales'),('clipped_units','orders_clipped'),('delivered_units','orders_delivered')]:
                assert abs(sum(np.asarray(r['info'][field]).sum() for r in trace)-e[metric])<1e-8
            assert abs(np.mean([r['info']['total_on_hand'] for r in trace])-e['mean_inventory'])<1e-10
            assert abs(np.mean([np.asarray(r['info']['lost_sales'])>0 for r in trace])-e['stockout_sku_day_fraction'])<1e-10
            assert all(r['info']['total_on_hand']<=(250 if scenario=='S8' else 500)+1e-8 for r in trace)
            demand=[(r['info']['demand'],r['info']['price']) for r in trace]
            if key[1] in noise:assert noise[key[1]]==demand
            else:noise[key[1]]=demand
            observed[key]=result;count+=1
            if key[1]==keys[0] and key[0] in ['main_211','pooled_211','independent_211','full7_5']:targets[key]=trace
        assert observed.keys()==required
        seen=set()
        for key,trace in episodes(out/f'{name}_replays.jsonl'):
            assert key not in seen and trace==targets[key];seen.add(key);replays+=1
        assert seen==targets.keys()
        for family,refs,coverage,rngseed in [('primary',['full7_5'],.975,20261023),('secondary',['pooled','independent'],.9875,20261024),('exploratory',['full7_5','projected','constant','ss'],.95,20261025)]:
            for ref in refs:
                for metric in ['reward_day','fill']:
                    x=[[observed[f'main_{s}',k][metric]-observed[f'{ref}_{s}' if ref in ['pooled','independent'] else ref,k][metric] for k in keys] for s in seeds]
                    check_effect(x,analysis[family][metric if family=='primary' else f'{ref}_{metric}'],coverage,rngseed)
        positive=all(x>0 for x in analysis['primary']['reward_day']['seed_means'])
        gate=positive and analysis['primary']['reward_day']['interval'][0]>0 and analysis['primary']['fill']['interval'][0]>=-.005
        assert analysis['gate']==gate and analysis['every_seed_reward_positive']==positive
        for label in labels:
            for metric in ['reward_day','fill']:
                assert abs(np.mean([observed[label,k][metric] for k in keys])-analysis['means'][label][metric])<1e-10
            x=np.array([28*observed[label,k]['reward_day'] for k in keys])
            distribution=analysis['distributions'][label];ret=distribution['episodic_return']
            for field,value in [('mean',x.mean()),('median',np.median(x)),('std',x.std(ddof=1)),('p05',np.quantile(x,.05))]:
                assert abs(ret[field]-value)<1e-8
            half=t.ppf(.975,49)*x.std(ddof=1)/np.sqrt(50)
            np.testing.assert_allclose(ret['ci95'],[x.mean()-half,x.mean()+half],atol=1e-8,rtol=0)
            success=distribution['success'];n_success=sum(observed[label,k]['fill']>=.95 for k in keys)
            assert success['n']==ret['n']==50 and success['successes']==n_success
            p=n_success/50;z=1.959963984540054;den=1+z*z/50
            center=(p+z*z/100)/den;half=z*np.sqrt(p*(1-p)/50+z*z/10000)/den
            np.testing.assert_allclose(success['ci95'],[max(0,center-half),min(1,center+half)],atol=1e-12,rtol=0)
            assert success['rate']==p
        all_values[name]=observed
        print(f'Verified {name}: actions, economics, pairing, intervals and replays.',flush=True)
    assert all_values['S2']==all_values['S6']==all_values['S4_e15']
    assert count==17000 and replays==40 and summary['counters']==dict(episodes=17040,days=477120)
    assert summary['status']=='complete' and summary['seconds']<=12000 and summary['training_updates']==0
    assert summary['primary_gate']==summary['analysis']['S2']['gate'] and summary['test_scoring']
    write_json(out/'verification.json',dict(status='passed',checked_episodes=count,exact_replays=replays,
        source_checkpoint_trace_hashes=True,independent_intervals=True,new_simulator_days=0,
        baseline_actions_checked=True,heuristic_rounding_ambiguous_sku_decisions=ambiguous,
        primary_gate=summary['primary_gate'],verifier_sha256=digest(__file__)))


def audit_baselines(archive):
    """Exercise baseline checks on an existing validation archive, no rollouts."""
    from scripts.train_compact_residual import INPUTS
    coefficients=np.asarray(json.loads(Path(INPUTS['center']).read_text())['coefficients'])
    counts={k:0 for k in ['projected','constant','ss']};ambiguous=0
    for key,trace in episodes(Path(archive)):
        if key[0] not in counts:continue
        for row in trace:
            ambiguous+=check_baseline(row,coefficients);counts[key[0]]+=1
    assert all(counts.values())
    print(json.dumps(dict(archive=str(archive),sha256=digest(archive),checked_days=counts,
        ambiguous_sku_decisions=ambiguous,new_simulator_days=0)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--output');g.add_argument('--audit-baselines');a=p.parse_args()
    if a.output:verify(a.output)
    else:audit_baselines(a.audit_baselines)
