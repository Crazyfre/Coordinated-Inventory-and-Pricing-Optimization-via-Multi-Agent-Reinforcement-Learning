"""Two matched supervised arms, archived training data only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.algos.structured_residual import StructuredResidualActor
from src.algos.local_feature_residual import LocalFeatureResidualActor
from scripts.diagnose_residual_imitation import physical_nll


def metrics(actor,x,y):
    with torch.no_grad():
        probs=torch.stack([d.probs for d in actor(x)],1)
        pred=probs.argmax(-1);p=probs.gather(-1,y[...,None]).squeeze(-1)
        center=actor.centers(x);low=center<2;middle=y-center==1
        result=dict(accuracy=float((pred==y).float().mean()),target_probability=float(p.mean()),
            lower_center_accuracy=float((pred[low]==y[low]).float().mean()),
            middle_offset_recall=float((pred[middle]==y[middle]).float().mean()))
    return result,probs.numpy()


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    source=Path('outputs/residual_imitation/seed11_v1');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    paths=dict(data=str(source/'datasets.npz'),original=str(source/'summary.json'),checkpoint=str(source/'fitted.pt'),
        runner=__file__,actor='src/algos/local_feature_residual.py',base='src/algos/structured_residual.py',protocol='RESIDUAL_FEATURE_PROTOCOL.md')
    hashes={k:sha(v) for k,v in paths.items()}
    original=json.loads(Path(paths['original']).read_text())
    assert hashes['data']==original['dataset_sha256'] and hashes['checkpoint']==original['checkpoint_sha256']
    (out/'protocol.json').write_text(json.dumps(dict(paths=paths,hashes=hashes,epochs=40,seed=11,lr=.0003,batch=128),indent=2))
    ds=np.load(paths['data']);spec=torch.load(paths['checkpoint'],weights_only=True)['spec']
    torch.set_num_threads(1)
    data={s:(torch.tensor(ds[s+'_x']),torch.tensor(ds[s+'_y'])) for s in ['fit','holdout']}
    args=(data['fit'][0].shape[1],spec['n'],spec['pipe_horizon'],spec['coefficients'])
    torch.manual_seed(11);control=StructuredResidualActor(*args)
    torch.manual_seed(11);augmented=LocalFeatureResidualActor(*args)
    assert all(torch.equal(v,augmented.body.state_dict()[k]) for k,v in control.body.state_dict().items())
    with torch.no_grad():
        for a,b in zip(control(data['fit'][0]),augmented(data['fit'][0])):assert torch.equal(a.probs,b.probs)
    generator=torch.Generator().manual_seed(11)
    orders=torch.stack([torch.randperm(len(data['fit'][0]),generator=generator) for _ in range(40)])
    np.save(out/'permutations.npy',orders.numpy())
    results={};predictions={}
    for label,actor in [('control',control),('augmented',augmented)]:
        optimizer=torch.optim.Adam(actor.parameters(),lr=.0003);history=[];x,y=data['fit']
        for epoch,order in enumerate(orders):
            losses=[]
            for batch in order.split(128):
                if time.perf_counter()-start>300:raise TimeoutError('feature comparison deadline')
                loss=physical_nll(actor,x[batch],y[batch]);assert torch.isfinite(loss)
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),.5);optimizer.step()
                losses.append(loss.item())
            history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
        final={}
        for split,(x,y) in data.items():final[split],predictions[label+'_'+split]=metrics(actor,x,y)
        ck=dict(variant=LocalFeatureResidualActor.variant if label=='augmented' else 'structured_residual_actor_only_v1',
            spec=spec,obs_dim=args[0],state_dict=actor.state_dict(),training='supervised_only')
        torch.save(ck,out/(label+'.pt'))
        results[label]=dict(history=history,metrics=final,parameters=sum(p.numel() for p in actor.parameters()),
            gate=bool(final['holdout']['accuracy']>=.99 and final['holdout']['target_probability']>=.99),checkpoint_sha256=sha(out/(label+'.pt')))
        print(label,json.dumps(results[label]['metrics']),flush=True)
    np.savez_compressed(out/'probabilities.npz',**predictions)
    assert all(sha(v)==hashes[k] for k,v in paths.items())
    summary=dict(paths=paths,hashes=hashes,results=results,initial_distributions_equal=True,
        permutations_sha256=sha(out/'permutations.npy'),probabilities_sha256=sha(out/'probabilities.npz'),
        new_episodes=0,ppo_updates=0,validation_used=False,seconds=time.perf_counter()-start)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
