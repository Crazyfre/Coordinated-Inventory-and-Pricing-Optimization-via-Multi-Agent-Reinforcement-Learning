"""Reuse a complete preserved test prefix and finish only the frozen missing grid."""
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from scripts.train_compact_residual import setup,INPUTS,digest,write_json
from scripts.discover_parameter_rl import episode
from scripts.verify_parameter_confirmation import episodes,check_episode
from scripts.verify_structured_final import verify,check_baseline
from scripts.report_structured_final import report
from src.eval.structured_final import analyze
from src.eval.parameter_review import episode_metrics
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env


def validate_prefix(audit):
    files=audit['files']
    if len(files)!=1 or files[0]['file']!='S1_traces.jsonl':raise ValueError('only audited S1 prefix supported')
    record=files[0]
    if record['partial_rows'] or record['malformed_lines']:raise ValueError('partial episode requires separate recovery')
    if not 0<record['complete_episodes']<1700:raise ValueError('incomplete S1 prefix required')
    return record


def run():
    source=Path('outputs/structured_final/evaluation_v1');out=Path('outputs/structured_final/evaluation_recovered_v1')
    audit_path=Path('outputs/structured_final/evaluation_interruption_audit.json')
    audit=json.loads(audit_path.read_text());prefix=validate_prefix(audit)
    manifest=json.loads((source/'frozen_manifest.json').read_text())
    assert digest(source/'frozen_manifest.json')==audit['frozen_manifest_sha256']
    assert digest(source/prefix['file'])==prefix['sha256']
    assert (source/prefix['file']).stat().st_size==prefix['complete_bytes']
    for path,h in manifest['hashes'].items():assert digest(path)==h
    for r in manifest['policies'].values():assert digest(r['path'])==r['sha256']
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(source/'frozen_manifest.json',out/'frozen_manifest.json')
    shutil.copyfile(source/prefix['file'],out/prefix['file'])
    recovery_files=[__file__,'STRUCTURED_EVALUATION_RECOVERY.md',str(audit_path),'scripts/audit_final_interruption.py']
    recovery_hashes={p:digest(p) for p in recovery_files}
    write_json(out/'recovery_manifest.json',dict(source=str(source),audit=audit,hashes=recovery_hashes,
        original_elapsed_seconds=None,recovery_cap_seconds=12000,complete_episodes_repeated=0))
    policies={label:np.array(r['parameters']) for label,r in manifest['policies'].items()}
    policies.update(full7_5=np.full(10,7.5),projected='projected',constant='constant',ss='ss')
    retained=audit['complete_episodes'];counters=dict(episodes=retained,days=28*retained)
    began=time.perf_counter();deadline=began+12000;stage='setup'
    try:
        torch.set_num_threads(1);_,reference=setup(211)
        calibration=load_identifiable_training_calibration(INPUTS['calibration'])
        coefficients=np.array(json.loads(Path(INPUTS['center']).read_text())['coefficients'])
        analyses={}
        for name,scenario,elasticity in manifest['conditions']:
            stage=name;rows=[];targets={};seen=set();noise={}
            trace_path=out/f'{name}_traces.jsonl'
            if name=='S1':
                for key,trace in episodes(trace_path):
                    p=policies[key[0]];check_episode(trace,None if isinstance(p,str) else p,1914)
                    for row in trace:check_baseline(row,coefficients)
                    assert key not in seen;seen.add(key);rows.append(episode_metrics(trace))
                    demand=[r['info']['demand'] for r in trace]
                    if key[1] in noise:assert noise[key[1]]==demand
                    else:noise[key[1]]=demand
                    if key[1]==manifest['keys'][0] and key[0] in ['main_211','pooled_211','independent_211','full7_5']:targets[key[0]]=trace
                assert len(seen)==retained
            env=make_research_env(scenario,split='test',start=1914,days=28,params=calibration,
                pricing_scope='shared',elasticity=elasticity)
            with trace_path.open('a' if name=='S1' else 'w') as handle:
                for key in manifest['keys']:
                    for label,p in policies.items():
                        if (label,key) in seen:continue
                        _,trace=episode(env,p,key,label,handle,counters,deadline,reference)
                        assert all(r['start']==1914 and r['info']['total_on_hand']<=(250 if scenario=='S8' else 500)+1e-8 for r in trace)
                        demand=[r['info']['demand'] for r in trace]
                        if key in noise:assert noise[key]==demand
                        else:noise[key]=demand
                        rows.append(episode_metrics(trace));seen.add((label,key))
                        if key==manifest['keys'][0] and label in ['main_211','pooled_211','independent_211','full7_5']:targets[label]=trace
                    write_json(out/'progress.json',dict(condition=name,completed_scenarios=key-manifest['keys'][0]+1,
                        counters=counters,retained_episodes=retained))
            assert len(seen)==1700
            result=analyze(rows);analyses[name]=result
            write_json(out/f'{name}_metrics.json',dict(rows=rows,analysis=result))
            with (out/f'{name}_replays.jsonl').open('w') as handle:
                for label,expected in targets.items():
                    p=np.array(manifest['policies'][label]['parameters']) if label!='full7_5' else np.full(10,7.5)
                    _,actual=episode(env,p,manifest['keys'][0],label,handle,counters,deadline,reference)
                    assert actual==expected
            print(f'{name}: frozen50-key matrix and4 replays complete',flush=True)
        assert counters==dict(episodes=17040,days=477120) and time.perf_counter()<=deadline
        for path,h in {**manifest['hashes'],**recovery_hashes}.items():assert digest(path)==h
        with (source/prefix['file']).open('rb') as old,(out/prefix['file']).open('rb') as new:
            while True:
                chunk=old.read(1024*1024)
                if not chunk:break
                assert chunk==new.read(len(chunk))
        assert digest(source/'frozen_manifest.json')==digest(out/'frozen_manifest.json')
        write_json(out/'summary.json',dict(status='complete',analysis=analyses,primary_gate=analyses['S2']['gate'],
            counters=counters,seconds=time.perf_counter()-began,exact_replays=40,test_scoring=True,training_updates=0,
            retained_episodes=retained,newly_simulated_episodes=17040-retained,original_elapsed_seconds=None,
            timing_scope='recovery segment only; original total unavailable',
            frozen_manifest_sha256=digest(out/'frozen_manifest.json'),
            trace_hashes={p.name:digest(p) for p in out.glob('*.jsonl')}))
        stage='archive_verification';verify(out)
        proof=json.loads((out/'verification.json').read_text())
        proof.update(exact_original_prefix=True,complete_episodes_repeated=0,retained_episodes=retained,
            newly_simulated_episodes=17040-retained,original_total_runtime_unavailable=True,
            recovery_runner_sha256=digest(__file__))
        write_json(out/'verification.json',proof)
        stage='report';report(out)
        with (out/'REPORT.md').open('a',encoding='utf-8') as handle:
            handle.write(f'\n\n## Evaluation interruption disclosure\n\nThe original evaluation stopped externally. Its {retained} complete episodes were retained byte-for-byte; no complete episode was repeated. Only {17040-retained} missing scored/replay episodes were newly simulated. The reported evaluation seconds measure the recovery segment only; original total elapsed compute is unavailable and original total-cap compliance is not claimed. Policies,keys,conditions and inference remain the original pre-test freeze.\n')
        print('Recovered final matrix verified and report generated; actual visual QA/release still required.',flush=True)
    except Exception as exc:
        write_json(out/'failed_status.json',dict(stage=stage,error=str(exc),counters=counters,
            seconds=time.perf_counter()-began,retries=0))
        raise


if __name__=='__main__':run()
