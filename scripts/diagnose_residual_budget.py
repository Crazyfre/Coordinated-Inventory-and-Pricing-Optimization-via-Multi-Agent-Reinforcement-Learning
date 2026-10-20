"""Fixed 560/2000-step checkpoints on one natural-sampling supervised trajectory."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.algos.local_feature_residual import LocalFeatureResidualActor
from scripts.diagnose_residual_imitation import physical_nll
from scripts.diagnose_residual_features import metrics


def batches(count,updates,batch_size=128,seed=11):
    if min(count,updates,batch_size)<1:raise ValueError('positive sizes required')
    generator=torch.Generator().manual_seed(seed);done=0
    while done<updates:
        for batch in torch.randperm(count,generator=generator).split(batch_size):
            yield batch;done+=1
            if done==updates:return


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    root=Path('outputs/residual_features/seed11_v1');previous=json.loads((root/'summary.json').read_text())
    paths=dict(data=previous['paths']['data'],prior=str(root/'augmented.pt'),prior_summary=str(root/'summary.json'),
        actor='src/algos/local_feature_residual.py',base='src/algos/structured_residual.py',loss='scripts/diagnose_residual_imitation.py',
        metrics='scripts/diagnose_residual_features.py',runner=__file__,protocol='RESIDUAL_BUDGET_PROTOCOL.md')
    hashes={k:sha(v) for k,v in paths.items()}
    assert hashes['data']==previous['hashes']['data'] and hashes['prior']==previous['results']['augmented']['checkpoint_sha256']
    (out/'protocol.json').write_text(json.dumps(dict(paths=paths,hashes=hashes,updates=[560,2000],seed=11),indent=2))
    old=torch.load(paths['prior'],weights_only=True);spec=old['spec'];ds=np.load(paths['data'])
    data={k:(torch.tensor(ds[k+'_x']),torch.tensor(ds[k+'_y'])) for k in ['fit','holdout']}
    torch.set_num_threads(1);torch.manual_seed(11)
    actor=LocalFeatureResidualActor(old['obs_dim'],spec['n'],spec['pipe_horizon'],spec['coefficients'])
    optimizer=torch.optim.Adam(actor.parameters(),lr=.0003)
    order=list(batches(len(data['fit'][0]),2000));np.save(out/'batches.npy',torch.stack(order).numpy())
    results={};saved={};losses=[];x,y=data['fit']
    for update,batch in enumerate(order,1):
        if time.perf_counter()-start>180:raise TimeoutError('fixed budget deadline')
        loss=physical_nll(actor,x[batch],y[batch]);assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),.5)
        assert torch.isfinite(norm);optimizer.step();losses.append(float(loss))
        if update in [560,2000]:
            if update==560:assert all(torch.equal(v,old['state_dict'][k]) for k,v in actor.state_dict().items()),'560-step reproduction mismatch'
            torch.save(dict(variant=LocalFeatureResidualActor.variant,spec=spec,obs_dim=old['obs_dim'],
                state_dict=actor.state_dict(),optimizer=optimizer.state_dict(),update=update,training='supervised_budget_only'),out/f'update_{update}.pt')
            result={}
            for split,(xx,yy) in data.items():result[split],saved[f'{update}_{split}']=metrics(actor,xx,yy)
            results[str(update)]=result;print(update,json.dumps(result),flush=True)
    np.savez_compressed(out/'probabilities.npz',**saved)
    assert all(sha(v)==hashes[k] for k,v in paths.items())
    final=results['2000']['holdout']
    summary=dict(paths=paths,hashes=hashes,results=results,losses=losses,reproduced_560_exactly=True,
        gate=bool(final['accuracy']>=.99 and final['target_probability']>=.99),seconds=time.perf_counter()-start,
        artifact_hashes={p.name:sha(p) for p in out.iterdir() if p.suffix in ['.pt','.npy','.npz']},
        new_episodes=0,ppo_updates=0,validation_used=False,updates=2000)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
