"""One-shot predeclared fresh-seed confirmation; no retries or policy selection."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.algos.temporal_cover_ppo import TemporalCoverPPO
from src.eval.temporal_confirmation import SEEDS,KEYS,analyze
from scripts.pilot_temporal_cover_ppo import collect,STUDY
from scripts.train_compact_residual import INPUTS,digest,write_json
from scripts.train_clean_research import TrainingWindows
from src.experiments.research import make_research_env
from src.demand.identifiable_calibration import load_identifiable_training_calibration


def run(output):
    pilot=Path('outputs/structured_mappo/temporal_cover_ppo_v1')
    assert json.loads((pilot/'verification.json').read_text())['status']=='passed'
    assert json.loads((pilot/'summary.json').read_text())['analysis']['promising']==['episode_greedy']
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    covers=np.array(json.loads((STUDY/'sku_seed211_final.json').read_text())['mean'])
    paths={str(p) for folder in ['src','scripts'] for p in Path(folder).rglob('*.py')}
    paths|={str(p) for p in Path('data').rglob('*') if p.is_file() and ('aligned' in str(p) or 'sku' in p.name.lower())}
    paths|={'src/config/skus.json',INPUTS['calibration'],str(STUDY/'sku_seed211_final.json'),'TEMPORAL_CONFIRMATION_PROTOCOL.md',
        str(pilot/'verification.json'),str(pilot/'summary.json')}
    hashes={p:digest(p) for p in sorted(paths)}
    write_json(out/'manifest.json',dict(hashes=hashes,covers=covers,seeds=list(SEEDS),planned_episodes=11658,
        training_cap=6000,per_run_cap=900,assessment_cap=1800,total_cap=7800,test_scoring=False))
    torch.set_num_threads(1);start=time.perf_counter();counters=dict(episodes=0,days=0);timings={};stage='setup'
    try:
        env=TrainingWindows(make_research_env('S2',split='train',start=15,days=28,pricing_scope='per_sku',
            params=load_identifiable_training_calibration(INPUTS['calibration'])),28)
        finals={};stage='training'
        with (out/'training.jsonl').open('w') as handle:
            for j,seed in enumerate(SEEDS):
                for arm in ['episode','daily']:
                    label=f'{arm}_{seed}';run_start=time.perf_counter();deadline=min(start+6000,run_start+900)
                    model=TemporalCoverPPO(covers,arm,seed);torch.save(model.state(),out/f'{label}_initial.pt')
                    for u in range(64):
                        e,plans=model.plans();returns=[]
                        for i,plan in enumerate(plans):
                            result,_=collect(env,plan,2360000+j*10000+u*8+i//2,f'{label}_u{u}_p{i}',handle,counters,deadline)
                            returns.append(result['discounted_day'])
                        returns=np.array(returns).reshape(8,2)
                        metrics=model.update(plans,returns)
                        torch.save(dict(epsilon=e.tolist(),plans=plans.tolist(),returns=returns.tolist(),metrics=metrics,
                            state=model.state()),out/f'{label}_u{u:03d}.pt')
                        write_json(out/'progress.json',dict(stage=stage,run=label,updates=u+1,counters=counters))
                        if (u+1)%16==0:print(f'{label}: {u+1}/64',flush=True)
                    timings[label]=time.perf_counter()-run_start;assert timings[label]<=900
                    finals[label]=model.mean.detach().numpy().copy()
                    write_json(out/'training_timings.json',timings)
        training_seconds=time.perf_counter()-start;assert training_seconds<=6000
        assert counters==dict(episodes=10240,days=286720)
        frozen={f'{l}_u063.pt':digest(out/f'{l}_u063.pt') for l in finals}
        assert len(frozen)==10 and all(digest(p)==h for p,h in hashes.items())
        write_json(out/'assessment_freeze.json',dict(checkpoints=frozen,source_hashes=hashes,counters=counters,
            training_seconds=training_seconds))
        stage='assessment';assessment_start=time.perf_counter();deadline=min(start+7800,assessment_start+1800)
        rows=[];targets={};policies={'baseline':covers.astype(np.float32)[None],**{l:c[None] for l,c in finals.items()}}
        with (out/'assessment.jsonl').open('w') as handle:
            for key in KEYS:
                for label,plan in policies.items():
                    result,trace=collect(env,plan,key,label,handle,counters,deadline);rows.append(result)
                    if key==KEYS[0] and label!='baseline':targets[label]=trace
                write_json(out/'progress.json',dict(stage=stage,scenarios=key-KEYS[0]+1,counters=counters))
        with (out/'replays.jsonl').open('w') as handle:
            for label,expected in targets.items():
                _,trace=collect(env,policies[label],KEYS[0],label,handle,counters,deadline);assert trace==expected
        assert counters==dict(episodes=11658,days=326424)
        assert all(digest(p)==h for p,h in hashes.items()) and all(digest(out/p)==h for p,h in frozen.items())
        assessment_seconds=time.perf_counter()-assessment_start;elapsed=time.perf_counter()-start
        assert assessment_seconds<=1800 and elapsed<=7800
        result=analyze(rows)
        write_json(out/'summary.json',dict(status='complete',analysis=result,rows=rows,counters=counters,
            training_seconds=training_seconds,assessment_seconds=assessment_seconds,seconds=elapsed,timings=timings,
            test_scoring=False,archive_hashes={p.name:digest(p) for p in out.glob('*.pt')},
            trace_hashes={p.name:digest(p) for p in out.glob('*.jsonl')}))
        print(json.dumps(result),flush=True)
    except Exception as exc:
        write_json(out/'failed_status.json',dict(stage=stage,error=str(exc),counters=counters,seconds=time.perf_counter()-start));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    run(a.output)
    # Verification adds no simulator days and runs only after successful completion.
    from scripts.verify_temporal_confirmation import verify
    verify(a.output)
