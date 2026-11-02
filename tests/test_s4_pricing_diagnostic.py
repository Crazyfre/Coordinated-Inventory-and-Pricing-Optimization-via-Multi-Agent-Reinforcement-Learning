import pytest
from scripts.diagnose_s4_pricing import analyze


def test_s4_signal_requires_reward_and_service_and_complete_grid():
    metrics=['revenue','ordering_cost','holding_cost','stockout_cost','price_change_cost','lost_sales','clipped_units','stockout_sku_day_fraction']
    rows=[dict(label=f'price{p}',scenario=k,reward_day=100.+(p==2),fill=.7,**{m:0. for m in metrics})
          for p in range(5) for k in range(2330000,2330064)]
    assert analyze(rows)['promising_price_indices']==[2]
    for r in rows:
        if r['label']=='price2':r['fill']=.69
    assert analyze(rows)['promising_price_indices']==[]
    with pytest.raises(ValueError):analyze(rows[:-1])
