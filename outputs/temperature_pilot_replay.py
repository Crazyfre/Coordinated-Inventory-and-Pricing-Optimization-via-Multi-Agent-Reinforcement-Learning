"""Independent manual greedy/stochastic inference and temperature transitions."""
import copy
import json
import numpy as np
import torch
from outputs.conditional_credit_pilot_replay import choose,serial
from src.env.interface import JointAction


def replay_inference(model,env,key,advance,kind,*,greedy):
    env.reset(seed=key);trace=[];rewards=[]
    for day in range(1,29):
        price=choose(model,env,'pricing',day,greedy=greedy)[2] if env.pricing_due else None
        if price is not None:env.select_price(price)
        obs,_,action,_,_=choose(model,env,'ordering',day,greedy=greedy)
        _,reward,done,info=advance(env,JointAction(None,action),kind);assert done==(day==28)
        rewards.append(float(reward));trace.append(dict(day=day,obs=obs,price_actions=price,actions=action,reward=float(reward),info=info))
    return dict(rewards=rewards,trace=trace)


def strict(a,b):
    if isinstance(a,torch.Tensor):assert isinstance(b,torch.Tensor) and torch.equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k,v in a.items():strict(v,b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):strict(x,y)
    else:assert a==b


def temperature_transition(after,next_pre,value):
    expected=copy.deepcopy(after);expected['ordering_temperature']=value
    expected['model']['ordering.temperature'].fill_(value)
    strict(expected,next_pre)


def ledger_pair(stream,step,kind,context):
    claim=json.loads(next(stream));done=json.loads(next(stream));expect=dict(step=step+1,kind=kind,**context)
    assert claim==dict(event='claim',**expect) and done==dict(event='complete',**expect)
    return step+1
