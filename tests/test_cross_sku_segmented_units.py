import copy
import numpy as np
import pytest
from scripts import cross_sku_segmented_units as u


def test_synthetic_scan_branch_probes_replays_and_accounting(monkeypatch):
    class Env:
        base_price=np.ones(10)
        day=1
        _finished=False
        def pricing_observation(self):
            x=np.zeros(204);x[:10]=50;x[-20:-10]=2;return x
    def reset(env,key):env.day=1;env._finished=False
    def advance(env,covers,first,sku,action,counter,deadline):
        day=env.day;counter['days']+=1
        row=dict(day=day,reward=0.,pricing_obs=env.pricing_observation().tolist() if (day-1)%7==0 else None)
        env.day+=1;env._finished=day==28
        return row,env._finished
    def suffix(env,covers,sku,action,counter,deadline):
        rows=[]
        while not env._finished:rows.append(advance(env,covers,env.day,sku,action,counter,deadline)[0])
        return rows
    monkeypatch.setattr(u,'reset',reset);monkeypatch.setattr(u,'advance',advance);monkeypatch.setattr(u,'suffix',suffix)
    monkeypatch.setattr(u,'clone_environment',copy.deepcopy)
    monkeypatch.setattr(u,'resample_future',lambda env,keys:copy.deepcopy(env))
    monkeypatch.setattr(u,'snapshot',lambda env:dict(day=env.day))
    scans=[u.scan(Env(),np.ones(10),ep,999)[0] for ep in range(128)]
    frozen,days=u.freeze(scans,np.ones(10),['hash']*128,{})
    assert days==0 and frozen['planned_days']==21616+67*frozen['suffix_length_sum']
    result,days=u.branch(Env(),np.ones(10),scans[0],frozen['selections'][0],999)
    assert days==28+67*sum(29-d for d,i in scans[0]['selected'])
    assert len(result['states'])==2 and result['original_replay']==scans[0]['trace']
    for state in result['states']:
        assert len(state['draws'])==32 and len(state['replays'])==2
        sample=state['sample'];expected=[int(np.random.SeedSequence([2580000,0,sample['day'],sample['sku'],0,z]).generate_state(1,dtype=np.uint64)[0]) for z in [0,1]]
        assert state['draws'][0]['keys']==expected
    with pytest.raises(ValueError):u.freeze(scans[:-1],np.ones(10),['hash']*127,{})
    bad=copy.deepcopy(scans);bad[0]['key']=2540000
    with pytest.raises(ValueError):u.freeze(bad,np.ones(10),['hash']*128,{})


def test_fresh_assessment_grid_intervals_and_plan():
    items=u.plan();assert len(items)==388 and len(set(items))==388
    assert items[128]=='selection' and items[257]=='fit' and items[-2:]==['replays','summary']
    units=[]
    for e in range(128):
        records=[dict(result=dict(label=l,key=2590000+e,reward_day=91. if l=='cross' else 90.,fill=.7,price_counts=[0,0,0,0,40])) for l in u.LABELS]
        units.append(dict(episode=e,records=records))
    a=u.analyze(units);assert a['primary_pass'] and a['representation_pass'] and a['heuristic_pass']
    for r in ['baseline','local','heuristic']:
        assert a['effects'][f'{r}:reward_day']==dict(mean=1.,ci991667=[1.,1.])
    with pytest.raises(ValueError):u.analyze(units[:-1])
    bad=copy.deepcopy(units);bad[0]['records'][0]['result']['key']=2560000
    with pytest.raises(ValueError):u.analyze(bad)
