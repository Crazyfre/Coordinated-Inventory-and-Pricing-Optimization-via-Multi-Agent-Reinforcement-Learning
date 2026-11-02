"""Fixed prospective matched credit packages; no simulation on import."""
import json
from pathlib import Path
import numpy as np
from outputs.team_credit_mappo import ConcentratedWholeTeamMAPPO
from outputs.anchored_pricing_mappo import AnchoredPricingIntegratedMAPPO
from outputs.conditional_credit_pilot_support import (Accounting,plain_collect,calendar,load,episode_metrics,
    hash_value,CALIBRATION,COVERS,training_collect as conditional_collect)
from outputs.fresh_support_pilot_support import training_collect as integrated_collect
from src.algos.categorical_residual_mappo import CategoricalResidualMAPPO
from src.experiments.execution_journal import digest

ROOT=Path('outputs/direct_mappo/team_credit_batch_pilot_v1')
ARMS=('integrated','whole');TRAIN_KEYS=tuple(range(11480000,11481024));ASSESS_KEYS=tuple(range(11680000,11680096))
SEED=4301;TORCH_SEED=24201;UPDATES=32;BATCH=32
MAIN_DAYS=1699712;AUDIT_DAYS=1703296
MAIN_INTERNAL_SECONDS=21600;MAIN_EXTERNAL_SECONDS=21660
AUDIT_INTERNAL_SECONDS=28800;AUDIT_EXTERNAL_SECONDS=28860
INTERNAL_SECONDS=MAIN_INTERNAL_SECONDS;EXTERNAL_SECONDS=MAIN_EXTERNAL_SECONDS
PROTOCOL='TEAM_CREDIT_MAPPO_PILOT_PROTOCOL.md'


def policy(arm,env,covers,seed=SEED):
    if arm=='anchor':return CategoricalResidualMAPPO(env,covers,seed)
    if arm not in ARMS:raise ValueError('declared team-credit arm required')
    return (AnchoredPricingIntegratedMAPPO if arm=='integrated' else ConcentratedWholeTeamMAPPO)(env,covers,seed)


def training_collect(model,arm,env,key,reference,budget):
    if arm=='integrated':return integrated_collect(model,'integrated',env,key,reference,budget)
    if arm=='whole':return conditional_collect(model,'whole',env,key,reference,budget)
    raise ValueError('declared training arm required')


def critic_targets(arm,episodes):
    result={r:[] for r in ('pricing','ordering')}
    for ep in episodes:
        for role,rows in ep['rows'].items():
            if arm=='whole':
                actual,reference=ep['rewards'],ep['whole_reference_rewards']
                values=[sum(.99**j*(a-b)/100 for j,(a,b) in enumerate(zip(actual[r['start']:],reference[r['start']:]))) for r in rows]
            elif arm=='integrated':
                roots=[root for root in ep['roots'] if root['role']==role]
                values=[sum(e['probability']*e['residual_return'] for e in root['entries']) for root in roots]
            else:raise ValueError('declared credit targets required')
            result[role]+=np.asarray(values,dtype=np.float32).tolist()
    return result


def analysis(records,replicates=20000,seed=20280113):
    keys=[r['key'] for r in records['anchor']]
    assert len(keys)>1 and len(keys)==len(set(keys)) and set(records)=={'anchor',*ARMS}
    assert all([r['key'] for r in rows]==keys for rows in records.values())
    draws=np.random.default_rng(seed).integers(len(keys),size=(replicates,len(keys)));effects={}
    for treatment,control in (('integrated','anchor'),('whole','anchor'),('whole','integrated')):
        out={}
        for metric in ('reward_day','fill'):
            values=np.asarray([a[metric]-b[metric] for a,b in zip(records[treatment],records[control])])
            out[metric]=dict(mean=float(values.mean()),ci=np.quantile(values[draws].mean(1),[.05/12,1-.05/12]).tolist())
        out['gate']=bool(out['reward_day']['ci'][0]>0 and out['fill']['ci'][0]>=-.005 and
            (control!='anchor' or out['reward_day']['mean']>=.5))
        effects[f'{treatment}:{control}']=out
    learning={arm:effects[f'{arm}:anchor']['gate'] for arm in ARMS}
    return dict(means={a:{m:float(np.mean([r[m] for r in rows])) for m in ('reward_day','fill')}
        for a,rows in records.items()},effects=effects,learning_gates=learning,
        qualifying_arms=[arm for arm in ARMS if learning[arm]],any_learning_gate=any(learning.values()),
        all_learning_gates=all(learning.values()),peer_superiority_gate=effects['whole:integrated']['gate'],
        confidence=1-.05/6,bootstrap_replicates=replicates,bootstrap_seed=seed,trained_seeds=1,
        uncertainty='paired scenario resampling conditional on one trained seed',full_goal_achieved=False)


def source_inventory():
    old=json.loads(Path('outputs/direct_mappo/anchored_pricing_pilot_v1/manifest.json').read_text())
    assert all(digest(p)==h for p,h in old['hashes'].items())
    paths=set(old['hashes'])|{p.as_posix() for p in Path('outputs').glob('*team_credit*.py')}
    paths|={PROTOCOL,'outputs/direct_mappo/anchored_pricing_pilot_v1_audit/report.json',
        'outputs/direct_mappo/anchored_pricing_pilot_v1_team_target_inspection_v1.json','outputs/inspect_team_return_targets.py',
        'outputs/direct_mappo/team_credit_pilot_proof_v1.json','outputs/direct_mappo/team_credit_pilot_correctness_v1.xml'}
    proof=json.loads(Path('outputs/direct_mappo/team_credit_pilot_proof_v1.json').read_text())
    assert proof['status']=='PASS_SYNTHETIC' and proof['calibrated_days']==0 and proof['updates']==4
    assert all(digest(p)==h for p,h in proof['sources'].items())
    from xml.etree import ElementTree as ET
    tree=ET.parse('outputs/direct_mappo/team_credit_pilot_correctness_v1.xml').getroot()
    assert sum(int(s.attrib.get('tests',0)) for s in tree.iter('testsuite'))==3
    assert not list(tree.iter('failure')) and not list(tree.iter('error'))
    return {p:digest(p) for p in sorted(paths)}
