import numpy as np
from scripts.audit_repeated_futures import resample_future,snapshot,future_keys,summarize
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env


def test_future_resampling_preserves_shipments_observation_and_parent():
    params=load_identifiable_training_calibration('outputs/calibration_correction/identifiable_v2/train_only_nb_identifiable_v2.json')
    env=make_research_env('S2',split='train',start=15,params=params);env.reset(seed=960000);env.select_price(4)
    env.pipeline.place_orders(np.full(env.n,5.),day=env.day)
    before=snapshot(env);keys=future_keys(0,8,0)
    a=resample_future(env,keys);b=resample_future(env,keys);other=resample_future(env,future_keys(0,8,1))
    assert snapshot(a)==snapshot(b)==snapshot(other)==before
    for day in range(1,10):
        np.testing.assert_array_equal(a.pipeline.receive(day),b.pipeline.receive(day))
    assert snapshot(env)==before
    np.testing.assert_array_equal(a.sim(8),b.sim(8))
    assert not np.array_equal(a.sim(8),other.sim(8))
    leads_a=a.pipeline.place_orders(np.full(env.n,5.),8)
    leads_b=b.pipeline.place_orders(np.full(env.n,5.),8)
    np.testing.assert_array_equal(leads_a,leads_b)


def test_variance_decomposition_and_persistent_error():
    noise=np.arange(16,dtype=float)-7.5
    record=dict(seed=1,day=8,sku=0,actions=[0,1],reference_action=0,chosen_action=1,predictions=[0.,2.],
        draws=[dict(branches=[dict(value=100.),dict(value=102.+v)]) for v in noise])
    stats=summarize([record])
    assert stats['errors']['model']['mean_target_mse']==0
    assert stats['errors']['model']['persistent_mse_estimate']<0  # do not truncate
    assert stats['states'][0]['chosen_gain']==2
    assert stats['mean_rank_tied_best_rate']==1
    np.testing.assert_allclose(stats['errors']['model']['single_future_mse'],np.mean(noise**2))


def test_future_keys_are_reproducible_separate_streams():
    keys=[tuple(future_keys(0,8,rep)) for rep in range(16)]
    assert len(set(keys))==16
    assert all(a!=b for a,b in keys)
    assert keys[0]==tuple(future_keys(0,8,0))
