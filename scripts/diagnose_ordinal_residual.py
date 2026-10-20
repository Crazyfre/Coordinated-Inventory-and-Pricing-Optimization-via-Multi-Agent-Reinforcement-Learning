"""Single fixed-budget ordinal comparison; no validation or PPO."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.algos.ordinal_residual import OrdinalResidualActor
from src.algos.local_only_residual import LocalOnlyResidualActor
from scripts.diagnose_residual_features import metrics
from scripts.diagnose_residual_imitation import physical_nll


def class_metrics(prob,y,center):
    pred=prob.argmax(-1);pt=np.take_along_axis(prob,y[...,None],-1)[...,0]
    return {str(k):dict(count=int((y-center==k).sum()),
        recall=float((pred[y-center==k]==y[y-center==k]).mean()) if (y-center==k).any() else None,
        target_probability=float(pt[y-center==k].mean()) if (y-center==k).any() else None) for k in range(-2,3)}


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    root=Path('outputs/local_only/seed11_v1');old=json.loads((root/'summary.json').read_text())
    paths=dict(data=old['paths']['data'],batches=old['paths']['batches'],comparator=str(root/'summary.json'),
        checkpoint=str(root/'update_2000.pt'),reference=str(root/'probabilities.npz'),actor='src/algos/ordinal_residual.py',
        local_actor='src/algos/local_only_residual.py',base='src/algos/structured_residual.py',runner=__file__,protocol='ORDINAL_RESIDUAL_PROTOCOL.md')
    hashes={k:sha(v) for k,v in paths.items()}
    assert hashes['data']==old['hashes']['data'] and hashes['batches']==old['hashes']['batches']
    assert hashes['checkpoint']==old['artifact_hashes']['update_2000.pt'] and hashes['reference']==old['artifact_hashes']['probabilities.npz']
    (out/'protocol.json').write_text(json.dumps(dict(paths=paths,hashes=hashes,seed=11,updates=[560,2000]),indent=2))
    ds=np.load(paths['data']);ref=np.load(paths['reference']);ck=torch.load(paths['checkpoint'],weights_only=True);spec=ck['spec']
    data={s:(torch.tensor(ds[s+'_x']),torch.tensor(ds[s+'_y'])) for s in ['fit','holdout']}
    torch.set_num_threads(1);torch.manual_seed(11);args=(ck['obs_dim'],spec['n'],spec['pipe_horizon'],spec['coefficients'])
    actor=OrdinalResidualActor(*args);baseline=LocalOnlyResidualActor(*args)
    with torch.no_grad():
        for a,b in zip(actor(data['fit'][0]),baseline(data['fit'][0])):torch.testing.assert_close(a.probs,b.probs,rtol=0,atol=0)
    def save(name,opt=None,step=0):
        torch.save(dict(variant=actor.variant,spec=spec,obs_dim=ck['obs_dim'],state_dict=actor.state_dict(),
            optimizer=opt.state_dict() if opt else {},update=step,training='ordinal_supervised_only'),out/(name+'.pt'))
    save('initial');opt=torch.optim.Adam(actor.parameters(),lr=.0003);x,y=data['fit'];results={};arrays={};losses=[];classes={};refclasses={}
    for update,batch in enumerate(np.load(paths['batches']),1):
        if time.perf_counter()-start>180:raise TimeoutError('ordinal deadline')
        loss=physical_nll(actor,x[batch],y[batch]);assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),.5);assert torch.isfinite(norm)
        opt.step();losses.append(loss.item())
        if update in [560,2000]:
            save(f'update_{update}',opt,update);result={};cl={};rc={}
            for split,(xx,yy) in data.items():
                result[split],arrays[f'{update}_{split}']=metrics(actor,xx,yy)
                with torch.no_grad():c=actor.centers(xx).numpy()
                cl[split]=class_metrics(arrays[f'{update}_{split}'],yy.numpy(),c)
                rc[split]=class_metrics(ref[f'{update}_{split}'],yy.numpy(),c)
            results[str(update)]=result;classes[str(update)]=cl;refclasses[str(update)]=rc
            print(update,json.dumps(result),flush=True)
    np.savez_compressed(out/'probabilities.npz',**arrays);h=results['2000']['holdout']
    assert all(sha(v)==hashes[k] for k,v in paths.items())
    summary=dict(paths=paths,hashes=hashes,results=results,classes=classes,comparator=old['results'],comparator_classes=refclasses,
        losses=losses,thresholds=actor.thresholds().detach().tolist(),parameters=dict(ordinal=sum(p.numel() for p in actor.parameters()),local_only=sum(p.numel() for p in baseline.parameters())),
        gate=bool(h['accuracy']>=.99 and h['target_probability']>=.99),seconds=time.perf_counter()-start,
        artifact_hashes={p.name:sha(p) for p in out.iterdir() if p.suffix in ['.pt','.npz']},new_episodes=0,ppo_updates=0,validation_used=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
