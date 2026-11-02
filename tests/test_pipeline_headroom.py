import json
from pathlib import Path
import numpy as np
from src.experiments.pipeline_headroom import candidates,clone_env,fingerprint,prepare,advance,serial,assert_paired
from src.experiments.pipeline_headroom import fixed_rules
from src.experiments.research import make_research_env
from src.demand.identifiable_calibration import load_identifiable_training_calibration


def test_candidates_deduplicate_boundaries_and_preserve_all_labels():
    for value,count in ((0,11),(40,11),(5,21)):
        groups=candidates(np.full(10,value));assert len(groups)==count
        labels=[label for g in groups for label in g['labels']]
        assert len(labels)==len(set(labels))==21 and 'baseline' in groups[0]['labels']
        for g in groups:assert ((g['action']>=0)&(g['action']<=40)).all()


def test_fixed_rules_use_only_observed_timing_and_respect_maximum_orders():
    a=np.zeros(260,np.float32);a[:10]=20;a[0]=0;a[-10:]=5
    b=a.copy();a[10]=40;b[12]=40;q=np.zeros(10,dtype=np.int64)
    assert fixed_rules(a,q)==dict(timing_deficit='baseline',blind_deficit='baseline')
    assert fixed_rules(b,q)==dict(timing_deficit='sku0:+1',blind_deficit='baseline')
    q[0]=40;assert fixed_rules(b,q)==dict(timing_deficit='baseline',blind_deficit='baseline')


def test_real_research_clone_alias_isolation_and_paired_exogenous_draws():
    env=make_research_env('S2',split='train',start=1200,days=28,pricing_scope='per_sku',params=
        load_identifiable_training_calibration('outputs/calibration_correction/identifiable_v2/train_only_nb_identifiable_v2.json'))
    covers=np.asarray(json.loads(Path('outputs/rl_discovery/parameter_confirmation_v1/sku_seed211_final.json').read_text())['mean'])
    env.reset(seed=5179999)
    for _ in range(7):
        obs,price,q=prepare(env,covers);advance(env,obs,price,q)
    obs,price,q=prepare(env,covers);same=clone_env(env);other=clone_env(env);initial=fingerprint(other)
    parent=advance(env,obs,price,q)
    x,p,a=prepare(same,covers);replay=advance(same,x,p,a)
    assert serial(parent)==serial(replay) and fingerprint(env)==fingerprint(same)
    after_parent=fingerprint(env);x,p,a=prepare(other,covers);a[0]=0 if a[0]==40 else 40
    perturbed=advance(other,x,p,a)
    assert fingerprint(other)!=initial and fingerprint(env)==after_parent
    assert_paired([perturbed],[parent]);assert not np.array_equal(perturbed['action'],parent['action'])
