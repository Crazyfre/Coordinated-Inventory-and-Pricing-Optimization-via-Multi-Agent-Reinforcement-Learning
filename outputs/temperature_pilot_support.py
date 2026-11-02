"""Prospective constants/dispatch/statistics; no simulation or fitting on import."""
import json
from pathlib import Path
import numpy as np
import torch
from outputs.temperature_integrated_mappo import TemperatureIntegratedMAPPO
from outputs.conditional_credit_pilot_support import Accounting,plain_collect,calendar,load,episode_metrics,CALIBRATION,COVERS
from outputs.fresh_support_pilot_support import training_collect as integrated_collect
from src.algos.categorical_residual_mappo import CategoricalResidualMAPPO
from src.experiments.execution_journal import digest

ROOT=Path('outputs/direct_mappo/temperature_integrated_pilot_v1')
PROTOCOL='TEMPERATURE_MAPPO_PILOT_PROTOCOL.md'
ARMS=('fixed','annealed');SEED=4501;TORCH_SEED=26201;ASSESS_TORCH_BASE=27201000
UPDATES=32;BATCH=32;TRAIN_KEYS=tuple(range(12480000,12481024));ASSESS_KEYS=tuple(range(12680000,12680640))
ASSESS_POLICIES=('anchor','initial_stochastic','fixed_greedy','annealed_greedy','fixed_stochastic','annealed_stochastic')
FINAL_POLICIES=ASSESS_POLICIES[2:];ASSESS_DAYS=107520
MAIN_DAYS=2429952;AUDIT_DAYS=2433536
MAIN_INTERNAL_SECONDS=7200;MAIN_EXTERNAL_SECONDS=7260;AUDIT_INTERNAL_SECONDS=10800;AUDIT_EXTERNAL_SECONDS=10860
INTERNAL_SECONDS=MAIN_INTERNAL_SECONDS;EXTERNAL_SECONDS=MAIN_EXTERNAL_SECONDS
CONTRASTS=tuple((a,'anchor') for a in FINAL_POLICIES)+tuple((a,'initial_stochastic') for a in FINAL_POLICIES if a.endswith('stochastic'))+(
    ('annealed_greedy','fixed_greedy'),('annealed_stochastic','fixed_stochastic'))


def temperature(arm,batch):
    assert arm in ARMS and isinstance(batch,int) and 0<=batch<32
    return 1. if arm=='fixed' or batch<=7 else 1.-.75*(batch-7)/24


def policy(arm,env,covers,seed=SEED):
    if arm=='anchor':return CategoricalResidualMAPPO(env,covers,seed)
    return TemperatureIntegratedMAPPO(env,covers,seed,arm=arm)


def evaluation_spec(label):
    assert label in ASSESS_POLICIES
    if label=='anchor':return 'anchor','baseline.pt',True
    if label=='initial_stochastic':return 'fixed','fixed/batch_000.pt',False
    arm=label.split('_')[0];return arm,f'{arm}/final.pt',label.endswith('greedy')


def training_collect(model,arm,env,key,reference,budget):
    assert arm in ARMS
    return integrated_collect(model,'integrated',env,key,reference,budget)


def critic_targets(arm,episodes):
    assert arm in ARMS;out={r:[] for r in ('pricing','ordering')}
    for ep in episodes:
        for role in out:
            values=[sum(e['probability']*e['residual_return'] for e in root['entries']) for root in ep['roots'] if root['role']==role]
            out[role]+=np.asarray(values,dtype=np.float32).tolist()
    return out


def analysis(records,replicates=50000,seed=20280116):
    assert set(records)==set(ASSESS_POLICIES)
    keys=[r['key'] for r in records['anchor']];assert len(keys)>1 and len(keys)==len(set(keys))
    assert all([r['key'] for r in rows]==keys for rows in records.values())
    draws=np.random.default_rng(seed).integers(len(keys),size=(replicates,len(keys)));effects={}
    for treatment,control in CONTRASTS:
        out={}
        for metric in ('reward_day','fill'):
            values=np.array([a[metric]-b[metric] for a,b in zip(records[treatment],records[control])])
            out[metric]=dict(mean=float(values.mean()),ci=np.quantile(values[draws].mean(1),[.05/32,1-.05/32]).tolist())
        out['gate']=bool(out['reward_day']['ci'][0]>0 and out['fill']['ci'][0]>=-.005 and
            (control!='anchor' or out['reward_day']['mean']>=.5))
        effects[f'{treatment}:{control}']=out
    gates={a:bool(effects[a+':anchor']['gate'] and
        (not a.endswith('stochastic') or effects[a+':initial_stochastic']['gate'])) for a in FINAL_POLICIES}
    return dict(means={a:{m:float(np.mean([r[m] for r in rows])) for m in ('reward_day','fill')} for a,rows in records.items()},
        effects=effects,learning_gates=gates,qualifying_modes=[a for a in FINAL_POLICIES if gates[a]],
        any_learning_gate=any(gates.values()),all_learning_gates=all(gates.values()),
        peer_superiority_gates={m:effects[f'annealed_{m}:fixed_{m}']['gate'] for m in ('greedy','stochastic')},
        confidence=1-.05/16,bootstrap_replicates=replicates,bootstrap_seed=seed,trained_seeds=1,
        uncertainty='paired scenarios conditional on one trained seed and fixed policy RNG scheme',full_goal_achieved=False)


def source_inventory():
    prior=Path('outputs/direct_mappo/paired_state_prediction_v1/input_freeze.json')
    old=json.loads(prior.read_text())['sources'];assert all(digest(p)==h for p,h in old.items())
    names=[PROTOCOL,'outputs/temperature_integrated_mappo.py','outputs/temperature_pilot_support.py',
        'outputs/temperature_pilot_replay.py','outputs/run_temperature_pilot_main.py','outputs/verify_temperature_pilot.py',
        'outputs/test_temperature_pilot.py','outputs/run_temperature_pilot_sequence.py','outputs/report_temperature_pilot.py',
        'outputs/summarize_temperature_review.py','outputs/preflight_temperature_pilot.py',
        'outputs/direct_mappo/temperature_pilot_proof_v1.json','outputs/direct_mappo/temperature_pilot_correctness_v1.xml',
        str(prior),'outputs/direct_mappo/paired_state_prediction_v1_audit/report.json',
        'outputs/direct_mappo/paired_state_prediction_v1/interpretation_receipt.json']
    proof=json.loads(Path(names[11]).read_text());assert proof['status']=='PASS_SYNTHETIC' and proof['updates']==4 and proof['calibrated_days']==0
    assert all(digest(p)==h for p,h in proof['sources'].items())
    from xml.etree import ElementTree as ET
    tree=ET.parse(names[12]).getroot();assert sum(int(s.attrib.get('tests',0)) for s in tree.iter('testsuite'))==4
    assert not list(tree.iter('failure')) and not list(tree.iter('error'))
    return dict(old,**{p:digest(p) for p in names})
