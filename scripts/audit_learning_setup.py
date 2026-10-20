"""Code/archived-state audit only: no environment construction or rollouts."""
import json
from pathlib import Path
import hashlib
import numpy as np
from src.algos.rollout_correction import features
from src.algos.structured_imitation import InventoryPositionLinearPolicy
from scripts.diagnose_cover_search import CoverPolicy


def run():
    root=Path('outputs/final_rollout_study/final_v1');out=Path('outputs/learning_setup_audit/code_archive_v1')
    out.mkdir(parents=True,exist_ok=False)
    ck=json.loads((root/'averaged_checkpoint.json').read_text());n,h=ck['n'],ck['horizon']
    paths=[Path(__file__),Path('src/algos/rollout_correction.py'),Path('src/env/replenishment_env.py'),
           Path('src/env/joint_env.py'),Path('src/env/pipeline.py'),Path('src/env/inventory.py'),
           Path('src/env/reward.py'),Path('src/algos/mappo.py'),Path('src/demand/simulator.py'),
           Path('src/config/locked_config.py'),Path('Docs/2. Implementation Reconciliation v2.md')]
    center_path=Path('outputs/structured_imitation/seed11_v1/structured_policy.json');paths.append(center_path)
    center=InventoryPositionLinearPolicy.load(center_path,n,h);target=CoverPolicy(n,h,7.5)
    cases=[];days={};state_days={};pipeline_states=0;total_states=0
    for phase in ['fit','assessment']:
        for ep in range(64):
            path=root/f'{phase}_{ep:03d}.json';paths.append(path);data=json.loads(path.read_text())
            for row in data['originals'][0]['trace']:
                key=str(row['day']);d=days.setdefault(key,dict(decisions=0,nonzero=0,units=0))
                a=np.array(row['actions']);d['decisions']+=len(a);d['nonzero']+=int((a>0).sum());d['units']+=int(5*a.sum())
            for r in data['records']:
                x=np.array(r['state']['obs'],dtype=np.float32);sku=r['sku'];total_states+=1
                state_days[str(r['day'])]=state_days.get(str(r['day']),0)+1
                pipe=x[n:n+n*h].reshape(n,h)
                pipeline_states+=int(pipe.sum()>0)
                # Structural witness: reverse an observed own-SKU shipment schedule,
                # preserving all inventory-position features and all other inputs.
                y=x.copy();yp=y[n:n+n*h].reshape(n,h);yp[sku]=pipe[sku,::-1]
                if not np.array_equal(x,y) and len(cases)<3:
                    for delta in [-2,-1,0,1,2]:np.testing.assert_array_equal(features(x,n,h,sku,delta,r['day']),features(y,n,h,sku,delta,r['day']))
                    np.testing.assert_array_equal(center.predict(x),center.predict(y));np.testing.assert_array_equal(target.act(x)[0],target.act(y)[0])
                    cases.append(dict(seed=r['seed'],day=r['day'],sku=sku,original_pipeline=pipe[sku].tolist(),
                                      modified_pipeline=yp[sku].tolist(),identical_regression_features=True,
                                      identical_center_and_target=True,scope='synthetic observation witness, not simulated reachable trajectory or Q-value proof'))
    assert cases
    # Exact analytic horizon visibility from locked lead support, not fitted probability.
    horizon=[dict(day=d,lead_values_arriving_by_terminal=[l for l in range(2,8) if d+l<=28],
                  earliest_arrival=d+2,latest_arrival=d+7) for d in range(21,29)]
    summary=dict(archived_episodes=128,archived_states=total_states,state_day_counts=state_days,
                 states_with_any_pipeline=pipeline_states,feature_collision_witnesses=cases,
                 original_actions_by_day=days,horizon_visibility=horizon,
                 no_economic_effect_order_days=[27,28],zero_new_episodes=True,zero_fits=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (out/'manifest.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps({k:summary[k] for k in ['archived_episodes','archived_states','state_day_counts','states_with_any_pipeline','feature_collision_witnesses']},indent=2))
    print(json.dumps({d:days[d] for d in ['21','22','26','27','28']},indent=2))


if __name__=='__main__':run()
