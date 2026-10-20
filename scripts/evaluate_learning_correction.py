"""One bounded paired clean S2 informed-prior vs vanilla MAPPO experiment."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from src.demand.identifiable_calibration import validate_research_training_calibration
from src.algos.mappo import HeterogeneousMAPPO
from src.algos.forecast_prior import ForecastPriorMAPPO
from src.experiments.research import make_research_env, training_statistics_view
from src.demand.alignment import build_aligned_dataset
from scripts.train_clean_research import TrainingWindows
from scripts.diagnose_operations import replay, summarize_trace


def run(output, updates=64, episodes=5, seed=11,
        calibration="outputs/evaluation_research/calibration/train_only_nb.json"):
    if not 1<=updates<=128 or not 1<=episodes<=5:
        raise ValueError('bounded updates 1..128, validation episodes 1..5')
    torch.set_num_threads(1)
    started=time.perf_counter()
    deadline=started+570
    out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    calibration=Path(calibration)
    params=json.loads(calibration.read_text())
    validate_research_training_calibration(params)
    aligned=training_statistics_view(build_aligned_dataset())
    config=dict(variant='forecast-informed categorical prior, NOT vanilla MAPPO advantage',
        scenario='S2',seed=seed,updates=updates,days=28,training_steps_per_method=updates*28,
        validation_seeds=list(range(10000,10000+episodes)),validation_start=1886,
        calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
        torch_threads=1,held_out_test_used=False,canonical_reward_and_physics=True,
        grid=list(range(0,201,5)),cover_days=6,width=2,uniform_mass=.05,
        hidden=64,lr=.0003,epochs=4,target_kl=.02,reward_scale=100,
        train_only_random_windows=True,max_experiment_seconds=570)
    config['pricing_scope'] = 'shared'
    config['calibration_status'] = params['status']
    config['calibration_path'] = str(calibration)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    results={}
    for label,cls in [('control',HeterogeneousMAPPO),('forecast_prior',ForecastPriorMAPPO)]:
        if time.perf_counter()>deadline: raise TimeoutError('experiment budget exceeded')
        train=TrainingWindows(make_research_env('S2',split='train',start=15,days=28,params=params,aligned=aligned,pricing_scope='shared'),28)
        validation=make_research_env('S2',split='validation',start=1886,days=28,params=params,aligned=aligned,pricing_scope='shared')
        torch.manual_seed(seed)
        model=cls(train,normalize_inputs=True,target_kl=.02)
        before=[p.detach().clone() for p in model.replenishment.parameters()]
        def measure(agent,stage,save=True):
            summaries=[]
            traces=[]
            for evseed in config['validation_seeds']:
                if time.perf_counter()>deadline: raise TimeoutError('experiment budget exceeded')
                rows=replay(validation,lambda e,o:agent.replenishment.act(o,True)[0],
                            lambda o:int(agent.pricing.act(o,True)[0][0]),evseed,label+'_'+stage)
                traces.extend(rows)
                summaries.append(summarize_trace(rows))
            if save:
                (out/(label+'_'+stage+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in traces))
            return summaries
        initial=measure(model,'initial')
        ts=time.perf_counter()
        history=model.train(updates,seed)
        if time.perf_counter()>deadline: raise TimeoutError('experiment budget exceeded')
        train_seconds=time.perf_counter()-ts
        if not all(np.isfinite(v) for r in history for v in r.values()):
            raise FloatingPointError('nonfinite training metrics')
        final=measure(model,'trained')
        path=out/(label+'.pt')
        model.save(path)
        restored=cls(validation,normalize_inputs=True,target_kl=.02)
        restored.load(path)
        assert final==measure(restored,'trained',False),'checkpoint replay differs'
        results[label]=dict(initial=initial,trained=final,history=history,
            training_seconds=train_seconds,training_windows=train.windows,
            checkpoint_roundtrip_verified=True,replenishment_parameter_l2_change=float(sum(
                (p.detach()-b).square().sum().item() for p,b in zip(model.replenishment.parameters(),before))**.5))
        (out/(label+'.json')).write_text(json.dumps(results[label],indent=2))
        print(label,train_seconds,flush=True)
    assert results['control']['training_windows']==results['forecast_prior']['training_windows']
    aggregates={label:{stage:{key:float(np.mean([r[key] for r in result[stage]]))
        for key in ['reward_day','fill','clips','mean_pipeline','core_fill']}
        for stage in ['initial','trained']} for label,result in results.items()}
    paired={key:[results['forecast_prior']['trained'][i][key]-results['control']['trained'][i][key]
        for i in range(episodes)] for key in ['reward_day','fill']}
    learning={key:[results['forecast_prior']['trained'][i][key]-results['forecast_prior']['initial'][i][key]
        for i in range(episodes)] for key in ['reward_day','fill']}
    summary=dict(aggregates=aggregates,paired_final_deltas=paired,prior_training_deltas=learning,
        elapsed_seconds=time.perf_counter()-started,held_out_test_used=False,
        prior_training_improved_reward=bool(np.mean(learning['reward_day'])>0),
        prior_beats_control_reward=bool(np.mean(paired['reward_day'])>0),
        service_target_met=aggregates['forecast_prior']['trained']['fill']>=.95)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/learning_correction/paired_s2_seed11')
    parser.add_argument('--updates',type=int,default=64)
    parser.add_argument('--episodes',type=int,default=5)
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--calibration',default='outputs/evaluation_research/calibration/train_only_nb.json')
    args=parser.parse_args()
    run(args.output,args.updates,args.episodes,args.seed,args.calibration)


if __name__=='__main__': main()
