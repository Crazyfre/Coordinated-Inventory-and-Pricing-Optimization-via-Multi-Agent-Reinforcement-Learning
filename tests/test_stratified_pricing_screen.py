import copy
import json
import numpy as np
import pytest
from src.algos.stratified_pricing import select_candidates
from scripts.screen_stratified_pricing import freeze_selection,assess


def test_selection_freeze_requires_complete_observable_selection(tmp_path):
    (tmp_path/'scan.jsonl').write_text('synthetic selection fixture\n')
    rows=[]
    for ep in range(128):
        observations={d:np.zeros(204).tolist() for d in [1,8,15,22]}
        rows.append(dict(episode=ep,key=2500000+ep,observations=observations,
            selected=[list(v) for v in select_candidates(observations,ep)]))
    with pytest.raises(ValueError):freeze_selection(tmp_path,rows[:-1],{},dict(days=3584))
    assert not (tmp_path/'selection_freeze.json').exists()
    bad=copy.deepcopy(rows);bad[0]['selected'][0]=[22,9]
    with pytest.raises(ValueError):freeze_selection(tmp_path,bad,{},dict(days=3584))
    bad=copy.deepcopy(rows);bad[0]['key']=1
    with pytest.raises(ValueError):freeze_selection(tmp_path,bad,{},dict(days=3584))
    with pytest.raises(ValueError):freeze_selection(tmp_path,rows,{},dict(days=3583))
    frozen=freeze_selection(tmp_path,rows,{},dict(days=3584))
    length=sum(29-day for r in rows for day,sku in r['selected'])
    assert frozen['planned_days']==12628+67*length<=492884
    assert json.loads((tmp_path/'selection_freeze.json').read_text())['suffix_length_sum']==length


def test_paired_assessment_intervals_and_service_guard():
    rows=[dict(label=l,key=k,reward_day=100.+(1 if l=='learned' else 0),fill=.7,
        price_counts=[0,0,0,0,40]) for k in range(2520000,2520064) for l in ['learned','baseline','heuristic']]
    result=assess(rows);assert result['primary_pass'] and result['heuristic_pass']
    assert result['effects']['baseline:reward_day']['ci9875']==[1.,1.]
    with pytest.raises(ValueError):assess(rows[:-1])
    with pytest.raises(ValueError):assess(rows+[rows[0]])
    for r in rows:
        if r['label']=='learned':r['fill']-=.006
    result=assess(rows);assert not result['primary_pass'] and not result['heuristic_pass']
