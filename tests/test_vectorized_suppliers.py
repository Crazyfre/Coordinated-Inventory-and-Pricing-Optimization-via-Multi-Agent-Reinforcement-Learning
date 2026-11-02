"""Bounded backend checks:7680supplier opportunities/112env steps, no updates."""
import copy
from dataclasses import replace
import numpy as np
from src.config.locked_config import get_locked_config
from src.experiments.phase4 import ScenarioPipeline
from src.experiments.vectorized_suppliers import VectorizedScenarioPipeline,VectorizedSupplierEnv
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env
from src.env.interface import JointAction


def test_exact_discrete_leads_rng_sparse_zero_uniform_and_fixed():
    cfg=get_locked_config().lead_time
    configs=(cfg,replace(cfg,distribution='uniform'),replace(cfg,min_days=4,max_days=4))
    for config in configs:
        for seed in range(32):
            old=ScenarioPipeline(10,config,100000+seed);new=VectorizedScenarioPipeline(10,config,100000+seed)
            new.rng=copy.deepcopy(old.rng);q_rng=np.random.default_rng(seed)
            for day in range(1,9):
                q=q_rng.integers(0,41,10).astype(float);q[(np.arange(10)+day)%3==0]=0
                if day==1:q[:]=0
                state=copy.deepcopy(old.rng.bit_generator.state)
                assert np.array_equal(old.place_orders(q,day),new.place_orders(q,day))
                assert old.rng.bit_generator.state==new.rng.bit_generator.state
                if not q.any():assert old.rng.bit_generator.state==state
                assert list(old._in_flight)==list(new._in_flight)
                assert all(np.array_equal(v,new._in_flight[k]) for k,v in old._in_flight.items())
                assert np.array_equal(old.receive(day),new.receive(day))
                assert np.array_equal(old.outstanding(),new.outstanding())
                assert np.array_equal(old.pending_by_remaining_days(day,7),new.pending_by_remaining_days(day,7))


def test_full_calibrated_traces_leads_and_rng_match_at_fixed_actions():
    params=load_identifiable_training_calibration('outputs/calibration_correction/identifiable_v2/train_only_nb_identifiable_v2.json')
    old=make_research_env('S2',split='train',start=15,days=28,pricing_scope='per_sku',params=params)
    new=VectorizedSupplierEnv(make_research_env('S2',split='train',start=15,days=28,pricing_scope='per_sku',params=params))
    for key in (5680100,5680101):
        a,_=old.reset(seed=key);b,_=new.reset(seed=key)
        assert all(np.array_equal(a[k],b[k]) for k in a)
        # Initial pipeline RNGs use OS entropy until the first positive order.
        new.pipeline.rng=copy.deepcopy(old.pipeline.rng)
        rng=np.random.default_rng(key)
        for day in range(1,29):
            price=rng.integers(0,5,10) if old.pricing_due else None
            q=rng.integers(0,41,10);q[(np.arange(10)+day)%3==0]=0
            a,r,d,i=old.step(JointAction(price,q));b,s,e,j=new.step(JointAction(price,q))
            assert r==s and d==e and all(np.array_equal(a[k],b[k]) for k in a)
            assert i.keys()==j.keys()
            assert all(np.array_equal(v,j[k]) for k,v in i.items())
            assert np.array_equal(old._last_lead_times,new._last_lead_times)
            assert old.pipeline.rng.bit_generator.state==new.pipeline.rng.bit_generator.state
            assert old.np_random.bit_generator.state==new.np_random.bit_generator.state
