"""Engineering-only real runner/worker and counted assessment equivalence."""
import json
import os
from pathlib import Path
import time
import numpy as np
import pytest
import torch
from scripts.recovery_campaign_runner import execute_unit
from scripts.recovery_campaign_plan import build
from scripts.recovery_campaign_science import select,assessment
from scripts.screen_cross_sku_pricing import reset,collect
from scripts.diagnose_conditional_pricing import suffix
from scripts.check_recovery_attempts import check
from src.experiments.recovery_execution import StepBudget
from src.experiments.supervised_work import SupervisedWork
from src.experiments.execution_journal import digest
from src.experiments.research import make_research_env
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from tests.test_recovery_campaign_plan import study
from tests.test_supervised_scientific_recovery import inputs


def environment(records):
    torch.set_num_threads(1)
    covers=np.array(json.loads(Path(records['ordering']['path']).read_text())['mean'])
    env=make_research_env('S2',split='train',start=15,pricing_scope='per_sku',
        params=load_identifiable_training_calibration(records['calibration']['path']))
    return env,covers


@pytest.mark.skipif(os.name!='nt',reason='Windows owned worker')
def test_real_runner_worker_scan_and_persistent_reset(tmp_path):
    config=study()
    config['inputs']=inputs()
    config['callback']=dict(module='scripts.recovery_campaign_dispatch',function='dispatch')
    config['source_hashes']={str(p):digest(p) for directory in ('src','scripts') for p in Path(directory).rglob('*.py')}
    root=tmp_path/'archive'
    results=[]
    with SupervisedWork(tmp_path/'scratch') as worker:
        for index in range(2):
            results.append(execute_unit(root,config,index,worker))
            if index==0:pid=worker.worker.pid
            else:assert worker.worker.pid==pid
    env,covers=environment(config['inputs'])
    counter={'days':0}
    for ep,result in enumerate(results):
        payload=json.loads(Path(result['artifact']).read_text())['payload']
        key=config['namespaces']['scan']+ep
        reset(env,key)
        trace=suffix(env,covers,0,4,counter,time.perf_counter()+120)
        assert payload['trace']==trace
        obs={r['day']:r['pricing_obs'] for r in trace if r['pricing_obs'] is not None}
        assert payload['selected']==select(obs,ep,env.base_price,config['namespaces']['scan'])
        assert payload['key']==key and result['days']==28
    report=check(root,build(config))
    assert report['completed_units']==2 and report['charged']['days']==56
    assert counter['days']==56  # Total real engineering steps112.


def test_counted_real_assessment_matches_direct_helpers():
    env,covers=environment(inputs())
    ns=dict(assessment=910600000)
    models={arm:dict(arm=arm,mean=[0.]*width,scale=[1.]*width,weights=[0.]*width)
            for arm,width in [('local',16),('cross',20)]}
    budget=StepBudget(112,time.perf_counter()+120,time.perf_counter)
    actual=assessment(env,covers,0,models,ns,budget)
    expected=[]
    counter={'days':0}
    for label in ('cross','local','baseline','heuristic'):
        result,trace=collect(env,covers,ns['assessment'],label,models,counter,time.perf_counter()+120)
        expected.append(dict(label=label,key=ns['assessment'],result=result,trace=trace))
    assert actual==dict(episode=0,records=expected)
    assert budget.days==counter['days']==112  # Total real engineering steps224.
