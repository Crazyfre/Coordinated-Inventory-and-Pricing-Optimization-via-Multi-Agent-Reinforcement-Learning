"""Fixed-budget local-only comparator; archived fit/holdout only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.algos.local_only_residual import LocalOnlyResidualActor
from src.algos.local_feature_residual import LocalFeatureResidualActor
from scripts.diagnose_residual_features import metrics
from scripts.diagnose_residual_imitation import physical_nll


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    root=Path('outputs/residual_budget/seed11_v1');old=json.loads((root/'summary.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    paths=dict(data=old['paths']['data'],comparator=str(root/'summary.json'),batches=str(root/'batches.npy'),
        comparator_probabilities=str(root/'probabilities.npz'),comparator_checkpoint=str(root/'update_2000.pt'),
        actor='src/algos/local_only_residual.py',base='src/algos/structured_residual.py',runner=__file__,protocol='LOCAL_ONLY_PROTOCOL.md')
    hashes={k:sha(v) for k,v in paths.items()}
    assert hashes['data']==old['hashes']['data']
    for key,name in [('batches','batches.npy'),('comparator_probabilities','probabilities.npz'),('comparator_checkpoint','update_2000.pt')]:assert hashes[key]==old['artifact_hashes'][name]
    (out/'protocol.json').write_text(json.dumps(dict(paths=paths,hashes=hashes,updates=[560,2000],seed=11),indent=2))
    ds=np.load(paths['data']);ck=torch.load(paths['comparator_checkpoint'],weights_only=True);spec=ck['spec']
    data={s:(torch.tensor(ds[s+'_x']),torch.tensor(ds[s+'_y'])) for s in ['fit','holdout']}
    torch.set_num_threads(1);torch.manual_seed(11)
    args=(ck['obs_dim'],spec['n'],spec['pipe_horizon'],spec['coefficients']);actor=LocalOnlyResidualActor(*args)
    comparator=LocalFeatureResidualActor(*args)
    with torch.no_grad():
        for a,b in zip(actor(data['fit'][0]),comparator(data['fit'][0])):assert torch.equal(a.probs,b.probs)
    def save(name,optimizer=None,update=0):
        torch.save(dict(variant=actor.variant,spec=spec,obs_dim=ck['obs_dim'],state_dict=actor.state_dict(),
            optimizer=optimizer.state_dict() if optimizer else {},update=update,training='local_only_supervised'),out/(name+'.pt'))
    save('initial');optimizer=torch.optim.Adam(actor.parameters(),lr=.0003)
    order=np.load(paths['batches']);assert order.shape==(2000,128)
    results={};probabilities={};losses=[];x,y=data['fit']
    for update,batch in enumerate(order,1):
        if time.perf_counter()-start>180:raise TimeoutError('local comparator deadline')
        loss=physical_nll(actor,x[batch],y[batch]);assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),.5)
        assert torch.isfinite(norm);optimizer.step();losses.append(loss.item())
        if update in [560,2000]:
            save(f'update_{update}',optimizer,update);result={}
            for split,(xx,yy) in data.items():result[split],probabilities[f'{update}_{split}']=metrics(actor,xx,yy)
            results[str(update)]=result;print(update,json.dumps(result),flush=True)
    np.savez_compressed(out/'probabilities.npz',**probabilities)
    assert all(sha(v)==hashes[k] for k,v in paths.items())
    h=results['2000']['holdout']
    summary=dict(paths=paths,hashes=hashes,results=results,comparator=old['results'],losses=losses,
        parameters=dict(local_only=sum(p.numel() for p in actor.parameters()),full_input=sum(p.numel() for p in comparator.parameters())),
        gate=bool(h['accuracy']>=.99 and h['target_probability']>=.99),initial_distributions_equal=True,
        seconds=time.perf_counter()-start,artifact_hashes={p.name:sha(p) for p in out.iterdir() if p.suffix in ['.pt','.npz']},
        new_episodes=0,ppo_updates=0,validation_used=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
