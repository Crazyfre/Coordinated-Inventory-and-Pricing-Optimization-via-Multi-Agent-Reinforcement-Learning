"""Fixed matched ridge fits differing only in reward-target averaging."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.stats import t
from scripts.train_rollout_correction import digest
from scripts.audit_rollout_predictions import rollout
from scripts.audit_repeated_futures import snapshot,resample_future
from scripts.diagnose_counterfactual_rollouts import clone_environment,decision,advance,check_branches
from scripts.diagnose_cover_search import CoverPolicy
from src.algos.rollout_correction import RolloutCorrection,features,fit_ridge
from src.algos.structured_imitation import InventoryPositionLinearPolicy
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env


def selected_days(ep):
    return [1+ep%7,8+ep%7,15+ep%7,22+ep%5,27+ep%2]


def collect(env,center,target,model,seed0,count,reps,namespace,deadline,label,episode_offset=0):
    records=[];originals=[]
    for local_ep,seed in enumerate(range(seed0,seed0+count)):
        ep=local_ep+episode_offset
        start=int(np.random.default_rng(np.random.SeedSequence([seed,314159])).integers(15,1859))
        env.sim.start_day_index=start;env.reset(seed=seed);original=[];episode=[]
        while not env._finished:
            c,ref=decision(env,center,target);day=env.day;chosen=model.action(env.replenishment_observation(),day)
            if day in selected_days(ep):
                changed=np.flatnonzero(chosen!=ref);assert len(changed)<=1
                sku=int(changed[0]) if len(changed) else (ep*5+selected_days(ep).index(day))%env.n
                actions=list(range(max(0,int(c[sku])-2),min(40,int(c[sku])+2)+1))
                before=snapshot(env);keys_before=[env.sim.scenario_seed,env.pipeline.scenario_seed]
                probe=rollout(clone_environment(env),center,target,model,chosen,'learned',deadline);draws=[]
                for rep in range(reps):
                    keys=[int(np.random.SeedSequence([namespace,ep,day,rep,s]).generate_state(1,dtype=np.uint64)[0]) for s in (0,1)]
                    bs=[]
                    for a in actions:
                        first=ref.copy();first[sku]=a
                        trace=rollout(resample_future(env,keys),center,target,model,first,'reference',deadline)
                        bs.append(dict(action=a,trace=trace,value=float(np.dot(.99**np.arange(len(trace)),[row['reward'] for row in trace]))))
                    baseline=next(b for b in bs if b['action']==ref[sku]);check_branches(bs,baseline['trace'])
                    if rep==0:
                        replay=rollout(resample_future(env,keys),center,target,model,ref,'reference',deadline)
                        assert replay==baseline['trace']
                    draws.append(dict(replicate=rep,keys=keys,branches=bs))
                assert snapshot(env)==before and [env.sim.scenario_seed,env.pipeline.scenario_seed]==keys_before
                episode.append(dict(seed=seed,start=start,day=day,sku=sku,actions=actions,reference_action=int(ref[sku]),
                    chosen_action=int(chosen[sku]),state=before,draws=draws,unresampled_probe=probe,replicate0_replay=replay))
            row,_=advance(env,chosen);original.append(row)
        for r in episode:assert r['unresampled_probe']==original[r['day']-1:]
        records.extend(episode);originals.append(dict(seed=seed,start=start,trace=original))
        if (ep+1)%4==0:print(f'{label}: {ep+1}/{count} scenarios',flush=True)
    return dict(records=records,originals=originals)


