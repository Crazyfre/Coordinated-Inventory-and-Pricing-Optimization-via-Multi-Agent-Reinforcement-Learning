import copy
import time
import numpy as np
import pytest
from scripts.recovery_campaign_science import CountedEnvironment, assemble_sample, analyze
from src.experiments.recovery_execution import StepBudget


def test_step_guard_precedes_actual_step():
    class Env:
        calls = 0
        def step(self, action):
            self.calls += 1
            return action
    raw = Env()
    budget = StepBudget(1,time.perf_counter()+10,time.perf_counter)
    env = CountedEnvironment(raw,budget)
    assert env.step(5) == 5
    with pytest.raises(RuntimeError):env.step(6)
    assert raw.calls == budget.days == 1


def test_assembly_complete_grid_and_recomputed_targets():
    state = dict(day=22,sku=3,before={'day':22},obs=[0.]*204)
    verification = dict(episode=0,states=[state])
    groups = []
    for group in range(8):
        draws = []
        for replica in range(4*group,4*group+4):
            branches = []
            for action in (0,4):
                trace = [dict(day=d,reward=float(replica if action==0 else 0)) for d in range(22,29)]
                branches.append(dict(action=action,trace=trace,value=sum(.99**i*r['reward'] for i,r in enumerate(trace))))
            draws.append(dict(replica=replica,branches=branches))
        groups.append(dict(episode=0,rank=0,group=group,day=22,sku=3,before=state['before'],draws=draws))
    sample = assemble_sample(verification,0,groups)
    np.testing.assert_allclose(sample['gains'],np.arange(32)*sum(.99**i for i in range(7)))
    with pytest.raises(ValueError):assemble_sample(verification,0,groups[:-1])
    bad = copy.deepcopy(groups)
    bad[0]['draws'][0]['branches'][0]['value'] += 1
    with pytest.raises(ValueError,match='return'):assemble_sample(verification,0,bad)


def test_unchanged_six_comparison_gates_and_fresh_assessment_keys():
    ns = dict(assessment=910300000,bootstrap=910400000)
    units = [dict(episode=e,records=[dict(result=dict(label=l,key=ns['assessment']+e,
            reward_day=91. if l=='cross' else 90.,fill=.7,price_counts=[0,0,0,0,40]))
            for l in ('cross','local','baseline','heuristic')]) for e in range(128)]
    result = analyze(units,ns)
    assert result['primary_pass'] and result['representation_pass'] and result['heuristic_pass']
    assert len(result['effects']) == 6
    assert result['effects']['baseline:reward_day']['ci991667'] == [1.,1.]
    bad = copy.deepcopy(units)
    bad[0]['records'][0]['result']['key'] = 2590000
    with pytest.raises(ValueError):analyze(bad,ns)
    for unit in units:
        unit['records'][0]['result']['fill'] = .69
    assert not analyze(units,ns)['primary_pass']
