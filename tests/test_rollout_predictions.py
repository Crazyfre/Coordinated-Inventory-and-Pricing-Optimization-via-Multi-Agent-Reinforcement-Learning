import numpy as np
from scripts.audit_rollout_predictions import interval,metrics


def test_interval_is_over_scenario_means():
    assert interval(np.ones(16)*3)==dict(mean=3.,ci95=[3.,3.])


def test_metrics_exclude_reference_and_compare_matched_continuations():
    records=[]
    for source in ['reference','learned']:
        for _ in range(64):
            records.append(dict(source=source,reference_action=0,chosen_action=1,predictions=[0.,2.],
                branches={'reference':[dict(action=0,value=10.),dict(action=1,value=12.)],
                          'learned':[dict(action=0,value=100.),dict(action=1,value=101.)]}))
    result=metrics(records)
    for source in result.values():
        assert source['reference']['nonreference_actions']==64
        assert source['reference']['rmse']==0
        assert source['reference']['zero_rmse']==2
        assert source['reference']['realized_tied_best_rate']==1
        assert source['learned']['rmse']==1
        assert source['continuation_delta']['mean']==-1
