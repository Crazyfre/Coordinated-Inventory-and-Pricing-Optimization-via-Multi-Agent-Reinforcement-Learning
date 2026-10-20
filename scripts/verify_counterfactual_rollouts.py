"""Recompute the saved branch audit without running new simulations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def verify(output):
    out = Path(output)
    manifest = json.loads((out/'manifest.json').read_text())
    for key, path in manifest['paths'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == manifest['hashes'][key]
    data = json.loads((out/'traces.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    assert len(data['records']) == 32
    refs = {r['seed']:r for r in data['references']}
    assert set(refs) == set(range(920000,920008))
    gains = []
    changed = 0
    for index, rec in enumerate(data['records']):
        seed = 920000 + index//4
        day = 1 + 7*(index%4)
        assert (rec['seed'],rec['day'],rec['sku']) == (seed,day,index%10)
        start = int(np.random.default_rng(np.random.SeedSequence([seed,314159])).integers(15,1859))
        assert rec['start'] == refs[seed]['start'] == start
        assert 15 <= start and start+27 <= 1885
        branches = rec['branches']
        assert [b['action'] for b in branches] == list(range(max(0,rec['center']-2),min(40,rec['center']+2)+1))
        ref = next(b for b in branches if b['action']==rec['reference_action'])
        assert ref['trace'] == refs[seed]['trace'][day-1:]
        for b in branches:
            rows = b['trace']
            assert [r['day'] for r in rows] == list(range(day,29))
            rewards = np.array([r['reward'] for r in rows])
            assert b['total'] == float(rewards.sum())
            assert b['discounted'] == float(np.dot(.99**np.arange(len(rows)),rewards))
            assert rows[0]['reward'] == ref['trace'][0]['reward']
            for i,(r,rr) in enumerate(zip(rows,ref['trace'])):
                assert r['demand']==rr['demand']
                assert 0 <= r['stock'] <= 500+1e-8
                assert all(0<=a<=40 for a in r['actions'])
                for sku,(a,ar) in enumerate(zip(r['actions'],rr['actions'])):
                    if i==0:
                        assert a == (b['action'] if sku==rec['sku'] else ar)
                    if a>0 and ar>0:
                        assert r['leads'][sku]==rr['leads'][sku]
        changed += any(b['total'] != ref['total'] for b in branches)
        gains.append(max(b['discounted'] for b in branches)-ref['discounted'])
    assert summary['states_with_changed_total_return']==changed
    assert summary['mean_hindsight_discounted_gain']==float(np.mean(gains))
    assert summary['branches']==sum(len(r['branches']) for r in data['records'])
    assert summary['branch_days']==sum(len(b['trace']) for r in data['records'] for b in r['branches'])
    assert summary['validity_gate'] and summary['nonzero_delayed_signal_gate']==(changed>0)
    assert summary['rl_updates']==0 and not summary['held_out_test_used']
    print('Verified hashes, training windows, support, traces, paired noise, returns and gates.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);verify(p.parse_args().output)
