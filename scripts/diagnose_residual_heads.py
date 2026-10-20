"""Matched shared-body versus separate-copy fitting on tiny balanced subsets."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.algos.local_feature_residual import LocalFeatureResidualActor
from scripts.diagnose_residual_tiny import select_rows,summarize


def masked_head_loss(distributions,targets,rows_by_head):
    return torch.stack([-distributions[i].log_prob(targets[:,i])[rows].mean()
                        for i,rows in rows_by_head.items()]).mean()


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source=Path('outputs/residual_imitation/seed11_v1');old=json.loads((source/'summary.json').read_text())
    paths=dict(data=str(source/'datasets.npz'),original=str(source/'summary.json'),checkpoint=str(source/'fitted.pt'),
        calibration=old['protocol']['paths']['calibration'],runner=__file__,actor='src/algos/local_feature_residual.py',
        base='src/algos/structured_residual.py',selector='scripts/diagnose_residual_tiny.py',protocol='RESIDUAL_HEAD_PROTOCOL.md')
    hashes={k:sha(v) for k,v in paths.items()}
    assert hashes['data']==old['dataset_sha256'] and hashes['checkpoint']==old['checkpoint_sha256']
    assert hashes['calibration']==old['protocol']['hashes']['calibration']
    ds=np.load(paths['data']);x=torch.tensor(ds['fit_x']);y=torch.tensor(ds['fit_y'])
    spec=torch.load(paths['checkpoint'],weights_only=True)['spec'];ids=json.loads(Path(paths['calibration']).read_text())['series_order']
    torch.set_num_threads(1);torch.manual_seed(11)
    initial=LocalFeatureResidualActor(x.shape[1],spec['n'],spec['pipe_horizon'],spec['coefficients'])
    with torch.no_grad():centers=initial.centers(x).numpy()
    selected={};availability={}
    for i in range(spec['n']):
        availability[str(i)]=[int(((centers[:,i]<2)&(ds['fit_y'][:,i]-centers[:,i]==k)).sum()) for k in [0,1,2]]
        if min(availability[str(i)])>=8:selected[i]=select_rows(centers,ds['fit_y'],i)
    if not selected:raise ValueError('no eligible heads')
    protocol=dict(paths=paths,hashes=hashes,seed=11,updates=2000,lr=.0003,sku_ids=ids,
        availability=availability,selected={str(k):v.tolist() for k,v in selected.items()},excluded=[i for i in range(spec['n']) if i not in selected])
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def save(actor,name):
        torch.save(dict(variant=LocalFeatureResidualActor.variant,spec=spec,obs_dim=x.shape[1],state_dict=actor.state_dict(),
            training='tiny_joint_separate_supervised_only'),out/(name+'.pt'))
    save(initial,'initial')
    all_probs={};results={};histories={}
    for arm in ['joint']+['separate_'+str(i) for i in selected]:
        heads=list(selected) if arm=='joint' else [int(arm.split('_')[1])]
        rows=np.unique(np.concatenate([selected[i] for i in heads]))
        local={i:torch.tensor(np.searchsorted(rows,selected[i])) for i in heads}
        xx,yy=x[rows],y[rows];actor=copy.deepcopy(initial)
        assert all(torch.equal(v,actor.state_dict()[k]) for k,v in initial.state_dict().items())
        optimizer=torch.optim.Adam(actor.parameters(),lr=.0003);history=[]
        def measured():
            stats={}
            with torch.no_grad():dist=actor(xx)
            for i in heads:
                prob=dist[i].probs[local[i]].numpy()
                target=ds['fit_y'][selected[i],i];offsets=target-centers[selected[i],i]
                stats[str(i)]=summarize(prob,target,offsets)
            return stats
        for update in range(1,2001):
            if time.perf_counter()-started>300:raise TimeoutError('head comparison deadline')
            loss=masked_head_loss(actor(xx),yy,local);assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),.5)
            assert torch.isfinite(norm);optimizer.step()
            if update%250==0:history.append(dict(update=update,heads=measured(),gradient_norm=float(norm)))
        stats=measured();save(actor,arm)
        with torch.no_grad():dist=actor(xx)
        for i in heads:all_probs[arm+'_head_'+str(i)]=dist[i].probs[local[i]].numpy()
        results[arm]=dict(heads=stats,gates={i:bool(r['accuracy']>=.99 and r['target_probability']>=.99) for i,r in stats.items()})
        histories[arm]=history;print(arm,json.dumps(results[arm]),flush=True)
    np.savez_compressed(out/'probabilities.npz',**all_probs)
    assert all(sha(v)==hashes[k] for k,v in paths.items())
    summary=dict(protocol=protocol,results=results,histories=histories,
        joint_gate=all(results['joint']['gates'].values()),separate_gate=all(results['separate_'+str(i)]['gates'][str(i)] for i in selected),
        total_decisions=24*len(selected),unique_states=len(np.unique(np.concatenate(list(selected.values())))),
        total_optimizer_updates=2000*(1+len(selected)),seconds=time.perf_counter()-started,
        hashes={p.name:sha(p) for p in out.glob('*.pt')},probabilities_sha256=sha(out/'probabilities.npz'),
        new_episodes=0,validation_used=False,ppo_updates=0)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
