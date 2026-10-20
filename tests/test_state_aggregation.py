import time
import torch
import pytest
import numpy as np
from src.algos.mappo import Actor
from scripts.diagnose_state_aggregation import acceptance, fit


def test_equivalence_requires_own_state_agreement_and_initial_gain():
    base=[dict(seed=i,reward_day=10.,fill=.6,teacher_agreement=1.) for i in range(4)]
    initial=[dict(r,reward_day=0.) for r in base]
    assert acceptance(1.,base,base,initial)['passed']
    assert not acceptance(.98,base,base,initial)['passed']
    assert not acceptance(1.,[dict(r,teacher_agreement=.98) for r in base],base,initial)['passed']
    assert not acceptance(1.,base,base,base)['passed']
    assert not acceptance(1.,[dict(r,fill=.59) for r in base],base,initial)['passed']


def test_supervised_fitting_uses_explicit_rng_and_respects_deadline():
    torch.set_num_threads(1)
    torch.manual_seed(9)
    actor=Actor(3,[2],8)
    optimizer=torch.optim.Adam(actor.parameters(),lr=.01)
    x=torch.ones(12,3); y=torch.ones(12,1,dtype=torch.long)
    before=[p.detach().clone() for p in actor.parameters()]
    result=fit(actor,optimizer,x,y,2,torch.Generator().manual_seed(4),time.perf_counter()+20)
    assert result['optimizer_steps']==2
    assert any(not torch.equal(a,b) for a,b in zip(before,actor.parameters()))
    with pytest.raises(TimeoutError):
        fit(actor,optimizer,x,y,1,torch.Generator().manual_seed(4),time.perf_counter()-1)


def test_collection_labels_learner_visited_states_not_teacher_trajectories():
    from scripts.diagnose_teacher_ordering import measure
    class TinyEnv:
        n=1
        obs_pipe_horizon=1
        pricing_due=True
        def reset(self,seed):
            self.day,self.stock=0,0.
        def select_price(self,index):
            assert index==4
        def replenishment_observation(self):
            return np.array([self.stock,0.,10.],dtype=np.float32)
        def step(self,action):
            self.stock=float(action.replenishment_qty_idx[0])*5
            self.day+=1
            return None,0.,self.day==2,dict(sales=np.array([0.]),demand=np.array([1.]),price=np.array([1.2]))
    class Learner:
        def act(self,obs,greedy=True):
            return np.array([0]),0.
    _,_,learner_states,learner_targets=measure(TinyEnv(),Learner(),[1],'learner',collect=True)
    _,_,teacher_states,teacher_targets=measure(TinyEnv(),None,[1],'teacher',collect=True)
    assert learner_targets[1].tolist()==[12]
    assert teacher_targets[1].tolist()==[0]
    assert learner_states[1][0]==0 and teacher_states[1][0]==60
