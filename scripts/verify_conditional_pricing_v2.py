"""Archive-only weekly branch actions, split-future selection and accounting."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.config.sku_config import load_skus
from scripts.train_compact_residual import digest,write_json


def verify(output):
    out=Path(output);load=lambda p:json.loads((out/p).read_text())
    manifest,summary=load('manifest.json'),load('summary.json');assert summary['status']=='complete'
    for p,h in manifest['hashes'].items():assert digest(p)==h,p
    for p,h in summary['trace_hashes'].items():assert digest(out/p)==h
    calibration=next(p for p in manifest['hashes'] if p.endswith('train_only_nb_identifiable_v2.json'))
    order=json.loads(Path(calibration).read_text())['series_order'];lookup={s.series_id:s.base_price for s in load_skus()}
    base=np.array([lookup[s] for s in order]);covers=np.array(manifest['covers'],np.float32)
    total_days=0;originals={};stats=[]

    def audit(trace,day,sku,alternative,keys,start):
        nonlocal total_days
        assert [r['day'] for r in trace]==list(range(day,29));prices=None;value=0.
        for i,r in enumerate(trace):
            assert r['start']==start and [r['demand_seed'],r['supplier_seed']]==keys
            x=np.array(r['obs'],np.float32);pipe=x[10:80].reshape(10,7)
            a=np.ceil(np.maximum(covers.clip(0,20)*x[-10:]-(x[:10]+pipe.sum(1)),0)/5).clip(0,40).astype(int)
            np.testing.assert_array_equal(a,r['actions'])
            if (r['day']-1)%7==0:
                pa=np.full(10,4)
                if r['day']==day:pa[sku]=alternative
                np.testing.assert_array_equal(pa,r['price_actions']);assert r['pricing_obs'] is not None
                prices=base*np.array([.8,.9,1.,1.1,1.2])[pa]
            else:assert r['price_actions'] is None and r['pricing_obs'] is None
            np.testing.assert_allclose(prices,r['info']['price'],atol=1e-12,rtol=0)
            terms=r['info']['reward_terms']
            reward=terms['revenue']-sum(terms[k] for k in ['ordering_cost','holding_cost','stockout_cost','price_change_cost'])
            assert abs(reward-r['reward'])<1e-8 and r['info']['total_on_hand']<=500+1e-8
            value+=.99**i*r['reward']
        total_days+=len(trace);return sum(.99**i*r['reward'] for i,r in enumerate(trace))

    with (out/'originals.jsonl').open() as f:
        for ep,line in enumerate(f):
            record=json.loads(line);key=2310000+ep;assert record['episode']==ep and record['key']==key
            start=int(np.random.default_rng(np.random.SeedSequence([key,314159])).integers(15,1859))
            assert record['start']==start and record['trace']==record['replay']
            trace=record['trace'];keys=[trace[0]['demand_seed'],trace[0]['supplier_seed']]
            audit(trace,1,0,4,keys,start);audit(record['replay'],1,0,4,keys,start)
            originals[ep]=record
    assert len(originals)==16
    with (out/'states.jsonl').open() as f:
        for index,line in enumerate(f):
            r=json.loads(line);ep=index//4;day=1+7*(index%4);sku=(ep*4+index%4)%10
            assert (r['episode'],r['key'],r['day'],r['sku'])==(ep,2310000+ep,day,sku)
            before=r['state_before'];assert before==r['state_after'] and before['day']==day and before['pending'] is None
            original=originals[ep];start=original['start'];assert before['start']==start
            assert r['probe']==original['trace'][day-1:]
            audit(r['probe'],day,sku,4,r['original_keys'],start)
            np.testing.assert_array_equal(np.array(before['inventory'],np.float32),np.array(before['obs'],np.float32)[:10])
            np.testing.assert_array_equal(np.array(before['history'],np.float32).T.ravel(),np.array(before['obs'])[80:220])
            values=[];assert len(r['draws'])==4
            for rep,draw in enumerate(r['draws']):
                keys=[int(np.random.SeedSequence([2320000,ep,day,rep,s]).generate_state(1,dtype=np.uint64)[0]) for s in [0,1]]
                assert keys==draw['keys'];assert len(draw['branches'])==5;result=[]
                for action,b in enumerate(draw['branches']):
                    assert b['action']==action and b['initial']==before
                    row=b['trace'][0]
                    np.testing.assert_array_equal(np.array(row['obs'])[:220],np.array(before['obs'])[:220])
                    np.testing.assert_array_equal(np.array(row['obs'])[230:250],np.array(before['obs'])[230:250])
                    np.testing.assert_array_equal(row['info']['orders_delivered'],before['accepted'])
                    np.testing.assert_array_equal(row['info']['orders_clipped'],before['clipped'])
                    value=audit(b['trace'],day,sku,action,keys,start)
                    assert value==b['value'];result.append(value)
                values.append(result)
                if rep==0:
                    assert r['replica0_replay']==draw['branches'][4]['trace']
                    audit(r['replica0_replay'],day,sku,4,keys,start)
            values=np.asarray(values);selection=values[:2].mean(0);chosen=4
            for action in [0,1,2,3]:
                if selection[action]>selection[chosen]:chosen=action
            expected=dict(episode=ep,key=2310000+ep,day=day,sku=sku,chosen=chosen,
                selection_gain=float(selection[chosen]-selection[4]),
                transferred_gain=float(np.mean(values[2:,chosen]-values[2:,4])),
                action_mean_gains=(values-values[:,4,None]).mean(0).tolist(),
                action_gain_std=(values-values[:,4,None]).std(0,ddof=1).tolist(),
                hindsight_gain=float(values.mean(0).max()-values[:,4].mean()))
            assert expected==r['result'];stats.append(expected)
    assert len(stats)==64 and stats==summary['statistics']
    x=np.array([s['transferred_gain'] for s in stats]).reshape(16,4).mean(1)
    draws=np.random.default_rng(20261028).integers(16,size=(10000,16))
    bounds=np.quantile([x[d].mean() for d in draws],[.025,.975]);discounts=sum(s['chosen']!=4 for s in stats)
    result=summary['analysis'];assert abs(result['mean_transferred_discounted_gain']-x.mean())<1e-10
    np.testing.assert_allclose(bounds,result['ci95'],atol=1e-10,rtol=0)
    assert result['selected_discount_states']==discounts and result['promising_signal']==bool(bounds[0]>0 and discounts>=8)
    assert abs(result['mean_hindsight_gain']-np.mean([s['hindsight_gain'] for s in stats]))<1e-10
    assert total_days==summary['days']==25536 and summary['seconds']<=900 and not summary['test_scoring']
    assert (summary['resampled_suffixes'],summary['original_episodes'],summary['full_replays'],summary['suffix_replays'],summary['original_probes'])==(1280,16,16,64,64)
    write_json(out/'verification.json',dict(status='passed',states=64,resampled_suffixes=1280,days=total_days,
        actions_economics_snapshots_noise_selection_intervals_replays=True,new_simulator_days=0,
        promising_signal=result['promising_signal'],verifier_sha256=digest(__file__)))
    print('Conditional pricing signal verified.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);verify(p.parse_args().output)

