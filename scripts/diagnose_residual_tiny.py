"""One fixed 24-decision lower-boundary memorization diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.algos.local_feature_residual import LocalFeatureResidualActor


def select_rows(center,targets,sku,k=8):
    choices=[]
    for offset in [0,1,2]:
        candidates=np.flatnonzero((center[:,sku]<2)&(targets[:,sku]-center[:,sku]==offset))
        if len(candidates)<k:raise ValueError('insufficient boundary examples; no fit permitted')
        choices.extend(candidates[:k].tolist())
    return np.asarray(choices,dtype=np.int64)


def summarize(probabilities,targets,offsets):
    pred=probabilities.argmax(-1);p=probabilities[np.arange(len(targets)),targets]
    result=dict(accuracy=float((pred==targets).mean()),target_probability=float(p.mean()),
        nll=float(-np.log(p).mean()),classes={})
    for offset in [0,1,2]:
        mask=offsets==offset
        result['classes'][str(offset)]=dict(count=int(mask.sum()),accuracy=float((pred[mask]==targets[mask]).mean()),target_probability=float(p[mask].mean()))
    return result


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    source=Path('outputs/residual_imitation/seed11_v1');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    original=json.loads((source/'summary.json').read_text())
    paths=dict(data=str(source/'datasets.npz'),original=str(source/'summary.json'),checkpoint=str(source/'fitted.pt'),
        calibration=original['protocol']['paths']['calibration'],runner=__file__,
        actor='src/algos/local_feature_residual.py',base='src/algos/structured_residual.py',protocol='RESIDUAL_TINY_PROTOCOL.md')
    hashes={k:sha(v) for k,v in paths.items()}
    assert hashes['data']==original['dataset_sha256'] and hashes['checkpoint']==original['checkpoint_sha256']
    assert hashes['calibration']==original['protocol']['hashes']['calibration']
    spec=torch.load(paths['checkpoint'],weights_only=True)['spec']
    ids=json.loads(Path(paths['calibration']).read_text())['series_order'];sku=ids.index('CA_3_HOUSEHOLD_1_506')
    ds=np.load(paths['data']);x=torch.tensor(ds['fit_x']);y=ds['fit_y']
    torch.set_num_threads(1);torch.manual_seed(11)
    args=(x.shape[1],spec['n'],spec['pipe_horizon'],spec['coefficients'])
    actor=LocalFeatureResidualActor(*args)
    with torch.no_grad():center=actor.centers(x).numpy()
    indices=select_rows(center,y,sku);xx=x[indices];yy=torch.tensor(y[indices,sku]);offsets=y[indices,sku]-center[indices,sku]
    np.savez(out/'subset.npz',indices=indices,observations=xx.numpy(),targets=yy.numpy(),offsets=offsets)
    protocol=dict(paths=paths,hashes=hashes,seed=11,sku=sku,sku_id=ids[sku],indices=indices.tolist(),updates=2000,lr=.0003)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def save(name):
        torch.save(dict(variant=LocalFeatureResidualActor.variant,spec=spec,obs_dim=args[0],state_dict=actor.state_dict(),
            training='tiny_subset_single_head_supervised_only'),out/(name+'.pt'))
        with torch.no_grad():return actor(xx)[sku].probs.numpy()
    initial=save('initial');optimizer=torch.optim.Adam(actor.parameters(),lr=.0003);history=[]
    for update in range(1,2001):
        if time.perf_counter()-start>180:raise TimeoutError('tiny fitting deadline')
        loss=-actor(xx)[sku].log_prob(yy).mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),.5)
        assert torch.isfinite(norm);optimizer.step()
        if update%100==0:
            with torch.no_grad():prob=actor(xx)[sku].probs.numpy()
            record=dict(update=update,gradient_norm=float(norm),**summarize(prob,yy.numpy(),offsets));history.append(record)
            if update%500==0:print(json.dumps(record),flush=True)
    final=save('final');np.savez(out/'probabilities.npz',initial=initial,final=final)
    result=summarize(final,yy.numpy(),offsets)
    assert all(sha(v)==hashes[k] for k,v in paths.items())
    summary=dict(protocol=protocol,initial=summarize(initial,yy.numpy(),offsets),final=result,history=history,
        gate=bool(result['accuracy']>=.99 and result['target_probability']>=.99),seconds=time.perf_counter()-start,
        artifact_hashes={name:sha(out/name) for name in ['initial.pt','final.pt','subset.npz','probabilities.npz']},
        new_episodes=0,ppo_updates=0,holdout_used=False,validation_used=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print(json.dumps(dict(final=result,gate=summary['gate'],seconds=summary['seconds']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
