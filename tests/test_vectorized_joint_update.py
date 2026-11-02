"""84calibrated steps/2joint updates: optimized simulator cannot change learning."""
import numpy as np
import torch
from src.algos.reference_baseline_mappo import ReferenceBaselineMAPPO
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env
from src.experiments.vectorized_suppliers import VectorizedSupplierEnv
from scripts.train_compact_residual import serial


def test_stochastic_joint_rollout_and_paired_update_are_exact():
    torch.set_num_threads(1)
    params=load_identifiable_training_calibration('outputs/calibration_correction/identifiable_v2/train_only_nb_identifiable_v2.json')
    def make():return make_research_env('S2',split='train',start=15,days=28,pricing_scope='per_sku',params=params)
    old_env=make();fast_env=VectorizedSupplierEnv(make());covers=np.linspace(4,10,10)
    old=ReferenceBaselineMAPPO(old_env,covers,102);fast=ReferenceBaselineMAPPO(fast_env,covers,102)
    key=5680102
    reference=old.collect(old_env,key,greedy=True,training=False)['rewards']
    torch.manual_seed(71237);a=old.collect(old_env,key)
    torch.manual_seed(71237);b=fast.collect(fast_env,key)
    assert serial(a)==serial(b)
    a['reference_rewards']=reference.copy();b['reference_rewards']=reference.copy()
    assert old.update([a])==fast.update([b])
    assert all(torch.equal(v,fast.state_dict()[k]) for k,v in old.state_dict().items())
    for role in ('pricing','ordering'):
        x=old.actor_optimizers[role].state_dict();y=fast.actor_optimizers[role].state_dict()
        assert x['param_groups']==y['param_groups']
        assert all(torch.equal(v,y['state'][i][k]) for i,state in x['state'].items() for k,v in state.items())
