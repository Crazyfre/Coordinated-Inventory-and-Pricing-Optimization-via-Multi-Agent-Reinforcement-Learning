import copy
import numpy as np
import pytest
from scripts.screen_cross_sku_pricing import freeze_selection,assess
from src.algos.cross_sku_rollout import selection
from scripts.train_compact_residual import digest


def test_selection_freeze_complete_and_hash_guard(tmp_path):
    x=np.zeros(204);x[:10]=50;x[-20:-10]=5;x[-10:]=0
    obs={d:x.tolist() for d in [1,8,15,22]};base=np.ones(10)
    rows=[dict(episode=e,key=2540000+e,observations=obs,selected=[list(v) for v in selection(obs,e,base)]) for e in range(128)]
    (tmp_path/'scan.jsonl').write_text('synthetic scan')
    source=tmp_path/'source';source.write_text('frozen');hashes={str(source):digest(source)}
    result=freeze_selection(tmp_path,rows,hashes,dict(days=3584),base)
    length=sum(29-d for r in rows for d,i in r['selected'])
    assert result['planned_days']==21616+67*length<=501872
    for bad in [rows[:-1],copy.deepcopy(rows)]:
        if len(bad)==128:bad[0]['key']=2500000
        with pytest.raises(ValueError):freeze_selection(tmp_path,bad,hashes,dict(days=3584),base)
    source.write_text('changed')
    with pytest.raises(ValueError):freeze_selection(tmp_path,rows,hashes,dict(days=3584),base)


def test_six_intervals_independent_reconstruction_and_separate_gates():
    rows=[]
    for key in range(2560000,2560128):
        for label in ['cross','local','baseline','heuristic']:
            gain=(key-2560000)/100+1 if label in ['cross','local'] else 0
            rows.append(dict(key=key,label=label,reward_day=90+gain,fill=.7,price_counts=[0,0,0,0,40]))
    a=assess(rows);assert a['primary_pass'] and a['heuristic_pass'] and not a['representation_pass']
    index={(r['label'],r['key']):r for r in rows};rng=np.random.default_rng(20261107)
    draws=[rng.integers(128,size=128) for _ in range(10000)]
    for ref in ['baseline','local','heuristic']:
        for metric in ['reward_day','fill']:
            values=np.array([index['cross',k][metric]-index[ref,k][metric] for k in range(2560000,2560128)])
            ci=np.quantile([values[d].mean() for d in draws],[.05/12,1-.05/12])
            np.testing.assert_allclose(ci,a['effects'][f'{ref}:{metric}']['ci991667'],rtol=0,atol=1e-12)
    with pytest.raises(ValueError):assess(rows[:-1])
    with pytest.raises(ValueError):assess(rows+[rows[0]])
    for r in rows:
        if r['label']=='cross':r['fill']=.694
    a=assess(rows);assert not a['primary_pass'] and not a['heuristic_pass']
