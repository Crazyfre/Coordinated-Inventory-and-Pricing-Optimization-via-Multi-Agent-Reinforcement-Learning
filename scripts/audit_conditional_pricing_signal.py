"""Read-only diagnosis of verified split-future pricing decisions."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.train_compact_residual import digest,write_json


def run(source,output):
    root=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    verification=json.loads((root/'verification.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    assert verification['status']=='passed'
    for p,h in summary['trace_hashes'].items():assert digest(root/p)==h
    rows=[]
    with (root/'states.jsonl').open() as handle:
        for line in handle:
            record=json.loads(line);chosen=record['result']['chosen'];sku=record['sku']
            values=np.array([[b['value'] for b in d['branches']] for d in record['draws']])
            selected=max([4,0,1,2,3],key=lambda a:values[:2,a].mean())
            assert selected==chosen
            second=max([4,0,1,2,3],key=lambda a:values[2:,a].mean())
            obs=np.array(record['probe'][0]['pricing_obs'])
            gains=values-values[:,4,None]
            economic={}
            for term in ['revenue','ordering_cost','holding_cost','stockout_cost','price_change_cost']:
                deltas=[]
                for draw in record['draws'][2:]:
                    a,b=draw['branches'][chosen]['trace'],draw['branches'][4]['trace']
                    deltas.append(sum(.99**i*(x['info']['reward_terms'][term]-y['info']['reward_terms'][term])
                                      for i,(x,y) in enumerate(zip(a,b))))
                economic[term]=float(np.mean(deltas))
            reconstructed=economic['revenue']-sum(economic[k] for k in economic if k!='revenue')
            assert abs(reconstructed-record['result']['transferred_gain'])<1e-8
            rows.append(dict(episode=record['episode'],day=record['day'],sku=sku,
                selected=chosen,second_half_best=second,ranking_agrees=chosen==second,
                stock=float(obs[sku]),forecast=float(obs[-20+sku]),
                weekly_stock_cover=float(obs[sku]/(1+7*obs[-20+sku])),
                selection_gain=record['result']['selection_gain'],
                transferred_gain=record['result']['transferred_gain'],economics=economic,
                gain_first_half=gains[:2].mean(0).tolist(),gain_second_half=gains[2:].mean(0).tolist()))
    assert len(rows)==64
    discounted=[r for r in rows if r['selected']!=4]
    first=np.array([r['gain_first_half'][:4] for r in rows]).ravel()
    second=np.array([r['gain_second_half'][:4] for r in rows]).ravel()
    result=dict(status='complete',states=64,discount_states=len(discounted),
        discount_positive_transfer=sum(r['transferred_gain']>0 for r in discounted),
        discount_negative_transfer=sum(r['transferred_gain']<0 for r in discounted),
        discount_zero_transfer=sum(r['transferred_gain']==0 for r in discounted),
        mean_selected_training_gain=float(np.mean([r['selection_gain'] for r in rows])),
        mean_selected_transfer_gain=float(np.mean([r['transferred_gain'] for r in rows])),
        all_state_ranking_agreement=float(np.mean([r['ranking_agrees'] for r in rows])),
        discount_state_ranking_agreement=float(np.mean([r['ranking_agrees'] for r in discounted])) if discounted else None,
        action_gain_half_correlation=float(np.corrcoef(first,second)[0,1]),
        mean_selected_economics={k:float(np.mean([r['economics'][k] for r in rows])) for k in rows[0]['economics']},
        rows=rows,new_simulator_days=0,
        provenance={p:digest(root/p) for p in ['summary.json','verification.json','states.jsonl']},
        limitation='Descriptive post-result audit on64 sampled states and four futures; no policy selection, fit or optimality claim.')
    write_json(out/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','provenance']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.source,a.output)
