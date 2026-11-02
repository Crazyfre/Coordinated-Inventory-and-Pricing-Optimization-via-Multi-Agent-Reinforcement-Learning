import copy
import numpy as np
import torch
from src.eval.critic_calibration import episode_data, fit, metrics
from scripts.verify_structured_critics import reconstruct_fit


def test_complete_discounted_targets_and_cadence():
    data=dict(rewards=[100.]*28,decisions=[dict(state=[0,1]) for _ in range(4)])
    result=episode_data(data,'pricing')
    np.testing.assert_allclose(result['targets'],[sum(.99**k for k in range(n)) for n in [28,21,14,7]])


def test_independent_normalized_fit_and_control_reconstruction():
    torch.set_num_threads(1);torch.manual_seed(1)
    critic=torch.nn.Sequential(torch.nn.Linear(2,64),torch.nn.Tanh(),torch.nn.Linear(64,1))
    other=copy.deepcopy(critic)
    data=[dict(states=[[float(i),.2],[float(i),.8]],targets=[2.*i+1,2.*i+4]) for i in range(4)]
    norm,history=fit(critic,data,epochs=4)
    assert reconstruct_fit(other,data,history)==norm
    assert norm['mean']==np.mean([1,4,3,6,5,8,7,10])


def test_time_control_can_prevent_false_pass():
    data=[dict(states=[[0,.2],[0,.8]],targets=[1.,4.]) for _ in range(64)]
    y=np.tile([1.,4.],64)
    result=metrics(data,dict(learned=y+.1,time=y,constant=np.full(128,2.5),original=np.zeros(128)))
    assert result['controllers']['learned']['explained_variance']>.99
    assert not result['passed']
