import pytest
from scripts.audit_pricing_intersection import branch_economics,TERMS


def test_signed_discounted_economic_reconstruction():
    a={k:0. for k in TERMS};a.update(revenue=10.,ordering_cost=3.,holding_cost=1.)
    b={k:0. for k in TERMS};b.update(revenue=5.,stockout_cost=2.)
    branches=[dict(action=3,value=6*1.99,trace=[dict(info=dict(reward_terms=a))]*2),
              dict(action=4,value=3*1.99,trace=[dict(info=dict(reward_terms=b))]*2)]
    d=branch_economics(branches)
    assert d['revenue']==5*1.99 and d['stockout_cost']==-2*1.99
    branches[0]['value']+=1
    with pytest.raises(AssertionError):branch_economics(branches)
