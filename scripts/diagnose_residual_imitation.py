"""One frozen-protocol supervised fit of the supported projected rule."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from scripts.diagnose_cover_search import CoverPolicy
from scripts.diagnose_residual_support import project_target
from scripts.diagnose_teacher_ordering import measure, compare
from scripts.train_clean_research import TrainingWindows
from src.algos.structured_imitation import InventoryPositionLinearPolicy
from src.algos.structured_residual import StructuredResidualMAPPO
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env


class ProjectedPolicy:
    def __init__(self, center, target):
        self.center, self.target = center, target

    def act(self, obs, greedy=True):
        return project_target(self.center.predict(obs), self.target.act(obs)[0]), 0.


def physical_nll(actor, x, y):
    return -torch.stack([d.log_prob(y[:, i]) for i, d in enumerate(actor(x))], -1).mean()


def agreement(actor, x, y):
    with torch.no_grad():
        predictions = torch.stack([d.probs.argmax(-1) for d in actor(x)], -1)
    return float((predictions == y).float().mean())


def run(output):
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter(); deadline = start + 570
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    paths = dict(center='outputs/structured_imitation/seed11_v1/structured_policy.json',
        selection='outputs/cover_search/bounded_v1/selection.json',
        calibration='outputs/calibration_correction/identifiable_v2/train_only_nb_identifiable_v2.json',
        runner=__file__, actor='src/algos/structured_residual.py', protocol='RESIDUAL_IMITATION_PROTOCOL.md')
    protocol = dict(paths=paths, hashes={k:digest(v) for k,v in paths.items()}, seed=11,
        fit_seeds=list(range(810000,810064)), holdout_seeds=list(range(810100,810116)),
        validation_seeds=list(range(90000,90020)), epochs=40, lr=.0003, batch=128,
        cover=json.loads(Path(paths['selection']).read_text())['selected']['cover'],
        rl_updates=0, held_out_test_used=False, max_seconds=570)
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    try:
        torch.set_num_threads(1)
        params = load_identifiable_training_calibration(paths['calibration'])
        train = TrainingWindows(make_research_env('S2', split='train', params=params, pricing_scope='shared'),28)
        val = make_research_env('S2', split='validation', params=params, pricing_scope='shared')
        center = InventoryPositionLinearPolicy.load(paths['center'], val.n, val.obs_pipe_horizon)
        teacher = ProjectedPolicy(center, CoverPolicy(val.n,val.obs_pipe_horizon,protocol['cover']))
        datasets = {}; rows = {}; traces = {}; accuracy = {}
        def evaluate(env, actor, seeds, label, greedy=True):
            r,t,x,_ = measure(env,actor,seeds,label,collect=True,greedy=greedy,deadline=deadline,trace_path=out/'traces.jsonl')
            y = np.asarray([teacher.act(o)[0] for o in x])
            x = np.asarray(x, dtype=np.float32)
            datasets[label+'_x'] = x; datasets[label+'_y'] = y
            rows[label], traces[label] = r,t
            accuracy[label] = float((np.asarray([v['actions'] for v in t])==y).mean())
            return torch.tensor(x),torch.tensor(y)
        x,y = evaluate(train,teacher,protocol['fit_seeds'],'fit')
        hx,hy = evaluate(train,teacher,protocol['holdout_seeds'],'holdout')
        windows = list(train.windows)
        torch.manual_seed(11)
        model = StructuredResidualMAPPO(val,paths['center'])
        optimizer = torch.optim.Adam(model.replenishment.parameters(),lr=.0003)
        history=[]
        initial = dict(fit=agreement(model.replenishment,x,y),holdout=agreement(model.replenishment,hx,hy))
        for epoch in range(40):
            losses=[]
            for batch in torch.randperm(len(x)).split(128):
                if time.perf_counter()>deadline: raise TimeoutError('bounded imitation deadline')
                loss=physical_nll(model.replenishment,x[batch],y[batch])
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite supervised loss')
                optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.replenishment.parameters(),.5);optimizer.step()
                losses.append(loss.item())
            history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),
                fit=agreement(model.replenishment,x,y),holdout=agreement(model.replenishment,hx,hy)))
            if (epoch+1)%10==0: print(history[-1],flush=True)
        model.save(out/'fitted.pt')
        for label,actor,greedy in [('center',center,True),('projected',teacher,True),
                ('greedy',model.replenishment,True),('sampled',model.replenishment,False)]:
            evaluate(val,actor,protocol['validation_seeds'],label,greedy)
        restored = StructuredResidualMAPPO(val,paths['center']); restored.load(out/'fitted.pt')
        for mode in ['greedy','sampled']:
            evaluate(val,restored.replenishment,protocol['validation_seeds'],'replay_'+mode,mode=='greedy')
            assert traces[mode]==[dict(r,label=mode) for r in traces['replay_'+mode]]
        comparisons={mode:{ref:compare(rows[mode],rows[ref]) for ref in ['projected','center']} for mode in ['greedy','sampled']}
        gates={mode:bool(history[-1]['holdout']>=.99 and accuracy[mode]>=.99 and
            comparisons[mode]['projected']['reward_ci95'][0]>=-1 and
            comparisons[mode]['projected']['fill_ci95'][0]>=-.005) for mode in comparisons}
        reference=[r['demand'] for r in traces['center']]
        assert all([r['demand'] for r in traces[label]]==reference for label in ['projected','greedy','sampled'])
        np.savez_compressed(out/'datasets.npz',**datasets)
        summary=dict(protocol=protocol,initial=initial,history=history,agreement=accuracy,comparisons=comparisons,
            gates=gates,imitation_gate=all(gates.values()),training_windows=windows,episode_rows=sum(rows.values(),[]),
            means={l:{k:float(np.mean([r[k] for r in rs])) for k in ['reward_day','fill']} for l,rs in rows.items()},
            episodes=len(sum(rows.values(),[])),trace_days=sum(map(len,traces.values())),seconds=time.perf_counter()-start,
            checkpoint_sha256=digest(out/'fitted.pt'),dataset_sha256=digest(out/'datasets.npz'),
            replay_verified=True,paired_demands_verified=True,rl_updates=0,held_out_test_used=False)
        assert all(digest(v)==protocol['hashes'][k] for k,v in paths.items())
        (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        print(json.dumps({k:summary[k] for k in ['means','agreement','gates','comparisons','seconds']},indent=2))
    except Exception as exc:
        (out/'failed_status.json').write_text(json.dumps(dict(status='failed_partial',error=str(exc))))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);run(p.parse_args().output)
