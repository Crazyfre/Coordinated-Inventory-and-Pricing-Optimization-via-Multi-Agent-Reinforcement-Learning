import copy
import pytest
from src.eval.temporal_confirmation import SEEDS,KEYS,analyze
from scripts.verify_temporal_confirmation import verify_analysis


def rows():
    result=[]
    for k in KEYS:
        result.append(dict(label='baseline',scenario=k,reward_day=100.,fill=.7))
        for s in SEEDS:
            for arm in ['episode','daily']:
                result.append(dict(label=f'{arm}_{s}',scenario=k,reward_day=101. if arm=='episode' else 100.,fill=.701))
    return result


def test_crossed_confirmation_and_independent_reconstruction():
    r=rows();a=analyze(r);verify_analysis({(x['label'],x['scenario']):x for x in r},a)
    assert a['primary_pass'] and a['matched_superiority_pass']
    altered=copy.deepcopy(a);altered['effects']['baseline:reward_day']['ci9875'][0]-=.1
    with pytest.raises(AssertionError):verify_analysis({(x['label'],x['scenario']):x for x in r},altered)


def test_missing_seed_duplicate_and_one_bad_seed_rejected():
    r=rows()
    with pytest.raises(ValueError):analyze(r[:-1])
    with pytest.raises(ValueError):analyze(r+[r[0]])
    for x in r:
        if x['label']=='episode_811':x['reward_day']=99.
    assert not analyze(r)['primary_pass']


def test_primary_and_matched_are_separate_and_service_matters():
    r=rows()
    for x in r:
        if x['label'].startswith('daily'):x['reward_day']=102.
    a=analyze(r);assert a['primary_pass'] and not a['matched_superiority_pass']
    for x in r:
        if x['label'].startswith('episode'):x['fill']=.69
    assert not analyze(r)['primary_pass']
