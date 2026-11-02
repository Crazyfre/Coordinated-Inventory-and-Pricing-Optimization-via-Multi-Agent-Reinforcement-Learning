import copy
import numpy as np
import pytest
from src.algos.rollout_pricing_rule import fit
from scripts.screen_rollout_pricing import assess
from scripts.verify_rollout_pricing import independent_fit,independent_analysis


def test_fitting_matches_independent_reconstruction_with_calendar_context():
    rng=np.random.default_rng(9);rows=[]
    for ep in range(16):
        for day in [1,8,15,22]:
            obs=rng.random(204).astype(np.float32);obs[:10]*=30;obs[-20:-10]*=4
            rows.append(dict(episode=ep,day=day,obs=obs.tolist(),advantages=rng.normal(size=(16,10)).tolist()))
    actual=fit(rows);expected=independent_fit(rows)
    for k,v in expected.items():
        if k=='scores':np.testing.assert_allclose(actual[k],v,rtol=0,atol=1e-12)
        else:assert actual[k]==v


def test_assessment_controls_service_guard_and_independent_intervals():
    rows=[dict(label=l,key=k,reward_day=100.+(1 if l=='learned' else 0),
        fill=.7,price_counts=[0,0,0,0,40]) for k in range(2460000,2460064) for l in ['learned','baseline','heuristic']]
    a=assess(rows);independent_analysis(rows,a);assert a['primary_pass'] and a['heuristic_pass']
    for r in rows:
        if r['label']=='learned':r['fill']-=.006
    a=assess(rows);independent_analysis(rows,a);assert not a['primary_pass'] and not a['heuristic_pass']
    bad=copy.deepcopy(a);bad['primary_pass']=True
    with pytest.raises(AssertionError):independent_analysis(rows,bad)
    with pytest.raises(ValueError):assess(rows[:-1])
    with pytest.raises(ValueError):assess(rows+[rows[0]])
