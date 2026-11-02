"""Fixed matched-prior pilot utilities; no simulation or launch on import."""
import json
from pathlib import Path
import numpy as np
import torch
from outputs.exploratory_credit_mappo import ExploratorySampledMAPPO,ExploratoryIntegratedMAPPO
from outputs.conditional_credit_pilot_support import (Accounting,CountedEnvironment,plain_collect,calendar,load,
    episode_metrics,hash_value,CALIBRATION,COVERS,training_collect as sampled_collect)
from src.algos.categorical_residual_mappo import CategoricalResidualMAPPO
from src.experiments.execution_journal import digest

ROOT=Path('outputs/direct_mappo/support_integrated_fresh_v1')
ARMS=('sampled','integrated');TRAIN_KEYS=tuple(range(9880000,9880256));ASSESS_KEYS=tuple(range(10080000,10080096))
SEED=4001;TORCH_SEED=21201;UPDATES=32;BATCH=8
MAIN_DAYS=430976;AUDIT_DAYS=434560;INTERNAL_SECONDS=7200;EXTERNAL_SECONDS=7260
PROTOCOL='FRESH_SUPPORT_INTEGRATED_MAPPO_PROTOCOL.md'


def policy(arm,env,covers,seed=SEED):
    if arm=='anchor':return CategoricalResidualMAPPO(env,covers,seed)
    return (ExploratorySampledMAPPO if arm=='sampled' else ExploratoryIntegratedMAPPO)(env,covers,seed)


def training_collect(model,arm,env,key,reference,budget):
    if arm=='sampled':return sampled_collect(model,'coordinate',env,key,reference,budget)
    before=torch.get_rng_state().clone();continuation=model.continuation;branch_before=budget.counts.get('branch',0)
    def counted(snapshot,role,action,**kwargs):
        previous=dict(budget.context);budget.context.update(root_role=role,root_day=snapshot.day);step=snapshot.step
        def counted_step(action):value=step(action);budget.done();return value
        snapshot.step=counted_step
        try:return continuation(snapshot,role,action,**kwargs)
        finally:del snapshot.step;budget.context=previous
    model.continuation=counted
    try:ep=model.collect(CountedEnvironment(env,budget),key,budget=budget,deadline=budget.deadline)
    finally:del model.continuation
    ep['sampled_reference_days']=ep['reference_days'];ep['reference_days']=budget.counts.get('branch',0)-branch_before
    ep.update(whole_reference_rewards=reference.copy(),policy_rng_before=before)
    assert torch.equal(torch.get_rng_state(),ep['actual_policy_rng'])
    return ep


def critic_targets(arm,episodes):
    result={r:[] for r in ('pricing','ordering')}
    for ep in episodes:
        for role,rows in ep['rows'].items():
            if arm=='sampled':values=[r['target'] for r in rows]
            else:
                roots=[r for r in ep['roots'] if r['role']==role]
                values=[sum(e['probability']*e['residual_return'] for e in root['entries']) for root in roots]
            # Save the exact float32 critic labels; independently verify each
            # label against reconstructed double returns before moment checks.
            result[role]+=np.asarray(values,dtype=np.float32).tolist()
    return result


def analysis(records,replicates=20000,seed=20280109):
    keys=[r['key'] for r in records['anchor']]
    assert len(keys)==len(set(keys)) and len(keys)>1
    assert set(records)=={'anchor','sampled','integrated'} and all([r['key'] for r in v]==keys for v in records.values())
    draws=np.random.default_rng(seed).integers(len(keys),size=(replicates,len(keys)));effects={}
    for control in ('anchor','sampled'):
        out={}
        for metric in ('reward_day','fill'):
            values=np.asarray([a[metric]-b[metric] for a,b in zip(records['integrated'],records[control])])
            out[metric]=dict(mean=float(values.mean()),ci=np.quantile(values[draws].mean(1),[.05/8,1-.05/8]).tolist())
        out['gate']=bool(out['reward_day']['ci'][0]>0 and out['fill']['ci'][0]>=-.005 and
                         (control!='anchor' or out['reward_day']['mean']>=.5));effects[control]=out
    return dict(means={a:{m:float(np.mean([r[m] for r in rows])) for m in ('reward_day','fill')} for a,rows in records.items()},
        effects=effects,all_pilot_gates=all(e['gate'] for e in effects.values()),confidence=.9875,bootstrap_replicates=replicates,
        bootstrap_seed=seed,trained_seeds=1,uncertainty='paired scenario resampling conditional on one trained seed',full_goal_achieved=False)


def source_inventory():
    old=json.loads(Path('outputs/direct_mappo/enumerated_credit_signal_v1/manifest.json').read_text())
    assert all(digest(p)==h for p,h in old['hashes'].items())
    paths=set(old['hashes'])|{p.as_posix() for p in Path('outputs').glob('*support_pilot*.py')}
    paths|={PROTOCOL,'outputs/summarize_fresh_matched_review.py','outputs/direct_mappo/support_integrated_pilot_v1_interruption_accounting_v1.json','outputs/direct_mappo/support_pilot_power_event_observation_v1.json','outputs/support_integrated_mappo.py','outputs/exploratory_credit_mappo.py',
        'outputs/test_support_integrated_mappo.py','outputs/test_exploratory_credit_mappo.py',
        'outputs/direct_mappo/support_integrated_proof_v1.json','outputs/direct_mappo/support_integrated_correctness_v1.xml',
        'outputs/direct_mappo/exploratory_credit_prior_proof_v1.json','outputs/direct_mappo/exploratory_credit_prior_correctness_v1.xml',
        'outputs/direct_mappo/support_pilot_replay_proof_v1.json','outputs/direct_mappo/support_pilot_replay_correctness_v1.xml',
        'outputs/direct_mappo/enumerated_credit_signal_v1_audit/report.json'}
    for name in ('support_integrated_proof_v1','exploratory_credit_prior_proof_v1','support_pilot_replay_proof_v1'):
        proof=json.loads(Path(f'outputs/direct_mappo/{name}.json').read_text());assert proof['status']=='PASS_SYNTHETIC' and proof['calibrated_days']==0
        assert all(digest(p)==h for p,h in proof['sources'].items())
    from xml.etree import ElementTree as ET
    for name,count in (('support_integrated_correctness_v1',2),('exploratory_credit_prior_correctness_v1',1),('support_pilot_replay_correctness_v1',3)):
        tests=ET.parse(f'outputs/direct_mappo/{name}.xml').getroot()
        assert sum(int(s.attrib.get('tests',0)) for s in tests.iter('testsuite'))==count
        assert not list(tests.iter('failure')) and not list(tests.iter('error'))
    return {p:digest(p) for p in sorted(paths)}
