import copy
import pytest
from scripts.recovery_campaign_plan import build, bind_request


def study():
    return dict(namespaces=dict(scan=910100000,future=910200000,assessment=910300000,bootstrap=910400000),
                seconds={k:120. for k in ('scan','selection','verify','group','fit','assessment','replays','summary')},
                caps=dict(days=600000,fit=7200.,assessment=600.,total=7800.),
                source_hashes={'fixture':'0'*64},inputs={'fixture':'fixed'},
                callback=dict(module='fixture',function='run'))


def test_complete_plan_dependencies_and_maximum_steps():
    config = study()
    plan = build(config)
    units = plan['units']
    assert len(units) == 2436
    assert len({u['item'] for u in units}) == 2436
    seen = set()
    for unit in units:
        assert set(unit['specification']['dependencies']) <= seen
        seen.add(unit['item'])
    assert sum(u['days'] for u in units) == 501872
    assert units[128]['item'] == 'selection'
    assert units[-131]['item'] == 'fit'
    assert sum(u['specification']['kind']=='group' for u in units) == 2048
    altered = copy.deepcopy(config)
    altered['namespaces']['future'] += 1
    assert build(altered)['identity'] != plan['identity']
    assert all(a['input_identity'] != b['input_identity'] for a,b in zip(units,build(altered)['units']))


def test_dependency_binding_rejects_missing_order_and_changed_unit():
    config = study()
    unit = build(config)['units'][129]
    records = [dict(item=item,path=item+'.json',sha256='0'*64,input_identity='fixed')
               for item in unit['specification']['dependencies']]
    assert bind_request(config,unit,records)['dependencies'] == records
    for bad in (records[:-1],list(reversed(records))):
        with pytest.raises(ValueError):bind_request(config,unit,bad)
    bad_unit = copy.deepcopy(unit)
    bad_unit['specification']['episode'] += 1
    with pytest.raises(ValueError):bind_request(config,bad_unit,records)
