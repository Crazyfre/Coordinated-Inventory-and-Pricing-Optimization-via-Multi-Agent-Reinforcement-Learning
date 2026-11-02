"""Single fixed-budget direct MAPPO integration pilot; no final-test scoring."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.algos.structured_mappo import StructuredMAPPO
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env
from src.experiments.execution_journal import digest,atomic_json
from scripts.train_compact_residual import serial

OUT=Path('outputs/direct_mappo/pilot_v1')
CALIBRATION='outputs/calibration_correction/identifiable_v2/train_only_nb_identifiable_v2.json'
ORDERING='outputs/rl_discovery/parameter_confirmation_v1/sku_seed211_final.json'


def main():
    began=time.perf_counter();deadline=began+840
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    paths={p.as_posix() for folder in ('src','scripts') for p in Path(folder).rglob('*.py')}
    paths|={CALIBRATION,ORDERING,'DIRECT_MAPPO_PILOT_PROTOCOL.md','src/config/skus.json',
        'data/processed/aligned_skus.csv','data/processed/aligned_skus.key.json'}
    hashes={p:digest(p) for p in sorted(paths)}
    atomic_json(OUT/'manifest.json',dict(hashes=hashes,training_seed=1701,updates=16,
        episodes_per_update=8,training_keys=[2970000,2970128],evaluation_keys=[2980000,2980016],
        expected_days=5376,internal_seconds=840,external_seconds=900,test_scoring=False))
    days=0
    def run(env,model,key,**kwargs):
        nonlocal days
        env.sim.start_day_index=int(np.random.default_rng(np.random.SeedSequence([key,314159])).integers(15,1859))
        if days+28>5376:raise RuntimeError('pilot day cap')
        result=model.collect(env,key,deadline=deadline,**kwargs)
        days+=28
        return result
    try:
        env=make_research_env('S2',split='train',start=15,days=28,pricing_scope='per_sku',
            params=load_identifiable_training_calibration(CALIBRATION))
        covers=np.asarray(json.loads(Path(ORDERING).read_text())['mean'])
        initial=StructuredMAPPO(env,covers,1701)
        model=StructuredMAPPO(env,covers,1701)
        torch.save(initial.checkpoint(),OUT/'initial.pt')
        history=[]
        for update in range(16):
            batch=[run(env,model,2970000+update*8+i) for i in range(8)]
            if time.perf_counter()>deadline:raise TimeoutError('pilot update deadline')
            metrics=model.update(batch)
            history.append(dict(update=update+1,**metrics,
                training_reward_day=float(np.mean([sum(e['rewards'])/28 for e in batch]))))
            atomic_json(OUT/'progress.json',dict(status='training',updates=update+1,days=days,
                elapsed_seconds=time.perf_counter()-began))
        torch.save(model.checkpoint(),OUT/'final.pt')
        atomic_json(OUT/'training_metrics.json',history)
        rows=[]
        with (OUT/'evaluation.jsonl').open('x',encoding='utf-8') as handle:
            for key in range(2980000,2980016):
                for label,agent,greedy in (('baseline',initial,True),('initial_sampled',initial,False),
                                           ('joint_greedy',model,True),('joint_sampled',model,False)):
                    torch.manual_seed(3100000+key)
                    episode=run(env,agent,key,greedy=greedy,training=False)
                    demand=sum(np.asarray(r['info']['demand']).sum() for r in episode['trace'])
                    sales=sum(np.asarray(r['info']['sales']).sum() for r in episode['trace'])
                    row=dict(key=key,label=label,reward_day=sum(episode['rewards'])/28,
                        fill=float(sales/demand) if demand else 1.)
                    handle.write(json.dumps(serial(dict(**row,trace=episode['trace'])),allow_nan=False)+'\n')
                    handle.flush();rows.append(row)
        if days!=5376:raise AssertionError('incomplete pilot')
        if any(digest(p)!=h for p,h in hashes.items()):raise RuntimeError('source/input changed during pilot')
        atomic_json(OUT/'report.json',dict(status='complete',days=days,
            elapsed_seconds=time.perf_counter()-began,rows=rows,
            means={label:{metric:float(np.mean([r[metric] for r in rows if r['label']==label]))
                for metric in ('reward_day','fill')} for label in ('baseline','initial_sampled','joint_greedy','joint_sampled')},
            claim='Developmental one-seed pilot; independent verification and matched controls still required.'))
    except BaseException as exc:
        atomic_json(OUT/'failure.json',dict(status='failed',error=f'{type(exc).__name__}: {exc}',
            completed_days=days,failed_episode_steps='unknown if interrupted',elapsed_seconds=time.perf_counter()-began))
        raise


if __name__=='__main__':main()
