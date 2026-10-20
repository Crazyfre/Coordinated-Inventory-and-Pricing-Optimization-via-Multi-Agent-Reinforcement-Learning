import pytest
from src.eval.structured_final import analyze,SEEDS,KEYS,CONDITIONS,BASELINES


def fixture():
    labels=[f'{c}_{s}' for c in CONDITIONS for s in SEEDS]+BASELINES
    return [dict(label=l,scenario=k,reward_day=k+ (2 if l.startswith('main_') else 0),fill=.7)
            for k in KEYS for l in labels]


def test_exact_paired_effect_and_incomplete_grid_rejected():
    rows=fixture();r=analyze(rows)
    assert r['gate'] and r['primary']['reward_day']['interval']==[2.,2.]
    assert r['secondary']['pooled_reward_day']['mean']==2
    with pytest.raises(ValueError):analyze(rows[:-1])
    rows[-1]=rows[0]
    with pytest.raises(ValueError):analyze(rows)


def test_service_and_negative_seed_cannot_be_hidden():
    rows=fixture()
    for r in rows:
        if r['label']=='main_310':r['reward_day']=r['scenario']-1
    assert not analyze(rows)['every_seed_reward_positive']
    for r in rows:
        if r['label'].startswith('main_'):
            r['reward_day']=r['scenario']+2;r['fill']=.69
    assert not analyze(rows)['gate']
