"""Continue an externally interrupted cohort using every completed episode."""
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from src.algos.structured_ablation import StructuredControl,pair_keys
from scripts.train_structured_final import SEEDS
from scripts.train_compact_residual import setup,digest,write_json
from scripts.discover_parameter_rl import episode
from scripts.verify_parameter_confirmation import episodes,check_episode


def run():
    source=Path('outputs/structured_final/training_v1');root=Path('outputs/structured_final/training_recovered_v1')
    root.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((source/'manifest.json').read_text());audit=json.loads(Path('outputs/structured_final/interruption_audit.json').read_text())
    for p,h in manifest['hashes'].items():assert digest(p)==h
    for p,h in audit['source_hashes'].items():assert digest(p)==h
    assert audit['saved_updates']==36 and audit['valid_daily_rows']==16380 and not audit['malformed_lines']
    history_old=json.loads((source/'main_266/history.json').read_text())
    reuse={}
    for key,trace in episodes(source/'main_266/traces.jsonl'):
        if int(key[0].split('_')[0][1:])>=36:reuse[key]=trace
    assert len(reuse)==9 and all(len(trace)==28 for trace in reuse.values())
    manifest['hashes'].update({p:digest(p) for p in ['STRUCTURED_TRAINING_RECOVERY.md',__file__,
        'scripts/verify_recovered_structured_training.py','outputs/structured_final/interruption_audit.json']})
    manifest['recovery']=dict(source=str(source),original_manifest_sha256=digest(source/'manifest.json'),
        retained_episodes=585,retained_days=16380,original_seed266_compute_seconds=None,
        timing_scope='resumed segment and subsequent runs only; original seed266 total time unavailable')
    write_json(root/'manifest.json',manifest)
    started=time.perf_counter();torch.set_num_threads(1);totals=dict(episodes=0,days=0);completed=[];stage='setup'
    try:
        env,reference=setup(266)
        for condition in ['main','pooled','independent']:
            for j,seed in enumerate(SEEDS):
                if condition=='main' and j<5:continue
                stage=f'{condition}_{seed}';out=root/stage;out.mkdir();began=time.perf_counter()
                deadline=min(began+1200,started+30000);learner=StructuredControl(condition,seed)
                first=condition=='main' and seed==266
                history=history_old.copy() if first else [];local=dict(episodes=585 if first else 0,days=16380 if first else 0)
                if first:
                    state=history[-1]['state']
                    for field in ['mean','sigma','m','v']:setattr(learner,field,np.array(state[field]))
                    learner.steps=state['steps'];learner.rng.bit_generator.state=state['rng']
                    shutil.copyfile(source/'main_266/traces.jsonl',out/'traces.jsonl')
                with (out/'traces.jsonl').open('a' if first else 'w') as handle:
                    for update in range(len(history),64):
                        e=learner.rng.normal(size=(8,len(learner.mean)));before=learner.mean.copy();records=[];differences=[]
                        for pair in range(8):
                            pk,mk=pair_keys(condition,j,update*8+pair);traces=[];scores={}
                            for sign,direction,key in [('plus',1,pk),('minus',-1,mk)]:
                                p=learner.expand(before+direction*learner.sigma*e[pair]);label=f'u{update}_p{pair}_{sign}'
                                if first and (label,key) in reuse:
                                    trace=reuse.pop((label,key));start=int(np.random.default_rng(np.random.SeedSequence([key,314159])).integers(15,1859))
                                    metrics=check_episode(trace,p,start)
                                    score=dict(label=label,scenario=key,**metrics)
                                else:score,trace=episode(env,p,key,label,handle,local,deadline,reference)
                                scores[sign]=score;traces.append(trace)
                            if condition!='independent':assert [r['info']['demand'] for r in traces[0]]==[r['info']['demand'] for r in traces[1]]
                            differences.append(scores['plus']['discounted_day']-scores['minus']['discounted_day'])
                            records.append(dict(plus_key=pk,minus_key=mk,**scores))
                        gradient=learner.update(e,differences)
                        history.append(dict(update=update+1,mean_before=before,epsilon=e,differences=differences,
                            gradient=gradient,records=records,state=learner.state()))
                        write_json(out/'history.json',history)
                        if (update+1)%16==0:print(f'{stage}: {update+1}/64',flush=True)
                assert local==dict(episodes=1024,days=28672) and time.perf_counter()<=deadline
                if first:assert not reuse
                write_json(out/'final.json',learner.state())
                result=dict(condition=condition,seed=seed,seconds=time.perf_counter()-began,counters=local,
                    recovered_prefix=first,original_compute_seconds=None if first else 0,
                    checkpoint_sha256=digest(out/'final.json'),trace_sha256=digest(out/'traces.jsonl'),history_sha256=digest(out/'history.json'))
                write_json(out/'summary.json',result);completed.append(result)
                for k in totals:totals[k]+=local[k]
                write_json(root/'progress.json',dict(completed_runs=len(completed),counters=totals,last_run=stage))
        for p,h in manifest['hashes'].items():assert digest(p)==h
        for p,h in audit['source_hashes'].items():assert digest(p)==h
        assert totals==dict(episodes=25600,days=716800)
        write_json(root/'summary.json',dict(status='complete',runs=completed,counters=totals,
            seconds=time.perf_counter()-started,test_scoring=False,
            newly_simulated=dict(episodes=25015,days=700420),retained=dict(episodes=585,days=16380),
            timing_scope=manifest['recovery']['timing_scope']))
        print('Recovered training complete; no complete episode repeated.',flush=True)
    except Exception as exc:
        write_json(root/'failed_status.json',dict(stage=stage,error=str(exc),completed_runs=len(completed),
            completed_counters=totals,seconds=time.perf_counter()-started))
        raise


if __name__=='__main__':run()
