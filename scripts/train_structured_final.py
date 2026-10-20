"""One-pass, predeclared final-cohort training; no test access or selection."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.algos.structured_ablation import StructuredControl,pair_keys
from scripts.discover_parameter_rl import episode
from scripts.train_compact_residual import setup,digest,write_json,INPUTS

SEEDS=[211,222,233,244,255,266,277,288,299,310]


def run(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    old=Path('outputs/rl_discovery/parameter_confirmation_v1')
    assert json.loads((old/'verification.json').read_text())['confirmed_gate']
    frozen=json.loads((old/'frozen_checkpoints.json').read_text())
    for seed in SEEDS[:5]:assert digest(old/f'sku_seed{seed}_final.json')==frozen[str(seed)]
    sources=['STRUCTURED_FINAL_STUDY_PROTOCOL.md','Docs/3. Structured RL study adoption.md',__file__,
        'scripts/verify_structured_training.py','src/algos/structured_ablation.py',
        'src/algos/parameter_policy_gradient.py','scripts/discover_parameter_rl.py',
        'scripts/train_compact_residual.py','scripts/train_clean_research.py',
        'src/experiments/research.py','src/experiments/phase4.py','src/env/joint_env.py',
        'src/env/replenishment_env.py','src/env/pipeline.py']+list(INPUTS.values())
    hashes={p:digest(p) for p in sources}
    write_json(root/'manifest.json',dict(hashes=hashes,seeds=SEEDS,updates=64,pairs=8,
        existing_main={str(s):dict(path=str(old/f'sku_seed{s}_final.json'),sha256=frozen[str(s)]) for s in SEEDS[:5]},
        run_cap_seconds=1200,total_cap_seconds=30000,test_scoring=False))
    torch.set_num_threads(1);started=time.perf_counter();counters=dict(episodes=0,days=0);completed=[]
    stage='setup'
    try:
        env,reference=setup(266)
        for condition in ['main','pooled','independent']:
            for j,seed in enumerate(SEEDS):
                if condition=='main' and j<5:continue
                stage=f'{condition}_{seed}';out=root/stage;out.mkdir()
                began=time.perf_counter();deadline=min(began+1200,started+30000)
                learner=StructuredControl(condition,seed);history=[];local=dict(episodes=0,days=0)
                with (out/'traces.jsonl').open('w') as handle:
                    for update in range(64):
                        e=learner.rng.normal(size=(8,len(learner.mean)));before=learner.mean.copy();records=[];differences=[]
                        for pair in range(8):
                            plus_key,minus_key=pair_keys(condition,j,update*8+pair);traces=[];scores={}
                            for sign,direction,key in [('plus',1,plus_key),('minus',-1,minus_key)]:
                                p=learner.expand(before+direction*learner.sigma*e[pair])
                                score,trace=episode(env,p,key,f'u{update}_p{pair}_{sign}',handle,local,deadline,reference)
                                scores[sign]=score;traces.append(trace)
                            if condition!='independent':
                                assert [r['info']['demand'] for r in traces[0]]==[r['info']['demand'] for r in traces[1]]
                            differences.append(scores['plus']['discounted_day']-scores['minus']['discounted_day'])
                            records.append(dict(plus_key=plus_key,minus_key=minus_key,**scores))
                        gradient=learner.update(e,differences)
                        history.append(dict(update=update+1,mean_before=before,epsilon=e,differences=differences,
                            gradient=gradient,records=records,state=learner.state()))
                        write_json(out/'history.json',history)
                        if (update+1)%16==0:print(f'{stage}: {update+1}/64',flush=True)
                assert local==dict(episodes=1024,days=28672)
                assert time.perf_counter()<=deadline
                write_json(out/'final.json',learner.state())
                result=dict(condition=condition,seed=seed,seconds=time.perf_counter()-began,counters=local,
                    checkpoint_sha256=digest(out/'final.json'),trace_sha256=digest(out/'traces.jsonl'),history_sha256=digest(out/'history.json'))
                write_json(out/'summary.json',result);completed.append(result)
                for k in counters:counters[k]+=local[k]
                write_json(root/'progress.json',dict(completed_runs=len(completed),counters=counters,last_run=stage))
        for p,h in hashes.items():assert digest(p)==h
        assert counters==dict(episodes=25600,days=716800)
        write_json(root/'summary.json',dict(status='complete',runs=completed,counters=counters,
            seconds=time.perf_counter()-started,test_scoring=False))
        print('All25 new training runs complete; no test scoring.',flush=True)
    except Exception as exc:
        write_json(root/'failed_status.json',dict(stage=stage,error=str(exc),completed_runs=len(completed),
            completed_counters=counters,seconds=time.perf_counter()-started))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
