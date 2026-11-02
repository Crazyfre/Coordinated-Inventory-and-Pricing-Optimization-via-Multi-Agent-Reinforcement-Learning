"""Read verified branches for daily/SKU response; zero new simulation."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.train_compact_residual import digest,write_json


def vectors(trace):
    rows=[]
    for r in trace:
        info=r['info'];d=np.array(info['demand']);sales=np.array(info['sales']);lost=np.array(info['lost_sales'])
        np.testing.assert_allclose(d-sales,lost,rtol=0,atol=1e-8)
        revenue=np.array(info['price'])*sales
        assert abs(revenue.sum()-info['reward_terms']['revenue'])<1e-8
        rows.append(dict(demand=d,sales=sales,lost_sales=lost,revenue=revenue,
            order_units=np.array(r['actions'])*5,accepted=np.array(info['orders_delivered']),
            clipped=np.array(info['orders_clipped']),forecast=np.array(r['obs'])[-10:]))
    return {k:np.array([r[k] for r in rows]) for k in rows[0]}


def run(source,output):
    root,out=Path(source),Path(output);summary=json.loads((root/'summary.json').read_text())
    assert json.loads((root/'verification.json').read_text())['status']=='passed'
    assert digest(root/'states.jsonl')==summary['trace_hash']
    result=[]
    with (root/'states.jsonl').open() as handle:
        for line in handle:
            record=json.loads(line)
            if record['episode'] not in [2,49,105,107]:continue
            totals=[];absolute=[]
            for draw in record['draws']:
                a,b=draw['branches'][0],draw['branches'][4]
                assert a['action']==0 and b['action']==4
                av,bv=vectors(a['trace']),vectors(b['trace'])
                totals.append({k:av[k]-bv[k] for k in av})
                absolute.append({k:np.abs(av[k]-bv[k]) for k in av})
            assert len(totals)==128
            mean={k:np.mean([t[k] for t in totals],axis=0) for k in totals[0]}
            disc=np.power(.99,np.arange(7))
            # Compare independently reconstructed revenue with verified summary.
            expected=next(s for s in summary['analysis']['states'] if s['episode']==record['episode'])
            assert abs(np.sum(mean['revenue']*disc[:,None])-expected['action_economics']['revenue'][0])<1e-8
            result.append(dict(episode=record['episode'],daily_sku_deltas={k:v.tolist() for k,v in mean.items()},
                sku4_week={k:float(v[:,4].sum()) for k,v in mean.items()},
                other_skus_week={k:float(np.delete(v,4,axis=1).sum()) for k,v in mean.items()},
                other_skus_absolute={k:float(np.mean([np.delete(t[k],4,axis=1).sum() for t in absolute])) for k in mean},
                sku4_daily_orders=mean['order_units'][:,4].tolist(),
                sku4_daily_accepted=mean['accepted'][:,4].tolist(),
                sku4_daily_lost=mean['lost_sales'][:,4].tolist(),
                day22_forecast_delta=float(mean['forecast'][0,4])))
    assert [r['episode'] for r in result]==[2,49,105,107]
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'audit.json',dict(status='complete',states=result,new_simulator_days=0,policy_fits=0,
        hashes={n:digest(root/n) for n in ['states.jsonl','summary.json','verification.json']},script_sha256=digest(__file__),
        limitation='Post-result paired trajectory description, not causal mediation or learned policy evidence. Weekly totals undiscounted; forecast sums are descriptive, not demand totals.'))
    print(json.dumps([{k:v for k,v in r.items() if k!='daily_sku_deltas'} for r in result]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.source,a.output)
