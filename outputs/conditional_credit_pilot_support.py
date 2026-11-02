"""Prospective pilot utilities; no CLI launch or training on import."""
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from outputs.linear_conditional_credit_mappo import LinearConditionalCreditMAPPO
from src.algos.categorical_residual_mappo import CategoricalResidualMAPPO
from src.experiments.execution_journal import digest
from scripts.train_compact_residual import serial

ROOT=Path('outputs/direct_mappo/conditional_credit_pilot_v1')
ARMS=('standard','whole','coordinate')
TRAIN_KEYS=tuple(range(8080000,8080256))
ASSESS_KEYS=tuple(range(8280000,8280096))
SEED=3601
TORCH_SEED=17201
UPDATES=32
BATCH=8
MAIN_DAYS=283136
AUDIT_DAYS=286720
INTERNAL_SECONDS=2700
EXTERNAL_SECONDS=2760
CALIBRATION='outputs/calibration_correction/identifiable_v2/train_only_nb_identifiable_v2.json'
COVERS='outputs/rl_discovery/parameter_confirmation_v1/sku_seed211_final.json'
PROTOCOL='CONDITIONAL_CREDIT_MAPPO_PILOT_PROTOCOL.md'


def hash_value(value):
    return hashlib.sha256(json.dumps(serial(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def load(path):
    return torch.load(path,map_location='cpu',weights_only=False)


def policy(arm,env,covers,seed=SEED):
    return (CategoricalResidualMAPPO(env,covers,seed) if arm in ('standard','anchor') else
            LinearConditionalCreditMAPPO(env,covers,seed,credit_mode=arm))


def calendar(env,key):
    env.sim.start_day_index=int(np.random.default_rng(np.random.SeedSequence([key,314159])).integers(15,1859))


class Accounting:
    def __init__(self,path,maximum,deadline):
        self.log=Path(path).open('x');self.maximum=maximum;self.deadline=deadline
        self.reserved=0;self.completed=0;self.pending=None;self.context={}
        self.counts={}
    def claim(self,kind,limit=float('inf')):
        if self.pending is not None or self.reserved>=self.maximum or time.perf_counter()>min(limit,self.deadline):
            raise RuntimeError('prospective step reservation/time bound')
        self.reserved+=1;self.pending=dict(step=self.reserved,kind=kind,**self.context)
        self.log.write(json.dumps(dict(event='claim',**self.pending))+'\n');self.log.flush()
    def done(self):
        assert self.pending is not None
        self.completed+=1;assert self.completed==self.reserved
        kind=self.pending['kind'];self.counts[kind]=self.counts.get(kind,0)+1
        self.log.write(json.dumps(dict(event='complete',**self.pending))+'\n');self.log.flush();self.pending=None
    def advance(self,env,action,kind):
        self.claim(kind);value=env.step(action);self.done();return value
    def close(self):self.log.close()


class CountedEnvironment:
    def __init__(self,env,budget):self.env=env;self.budget=budget
    def __getattr__(self,name):return getattr(self.env,name)
    def __deepcopy__(self,memo):return copy.deepcopy(self.env,memo)
    def step(self,action):value=self.env.step(action);self.budget.done();return value


def plain_collect(model,env,key,budget,kind,*,training=False,greedy=True):
    step=env.step
    def counted(action):budget.claim(kind);value=step(action);budget.done();return value
    env.step=counted
    try:return model.collect(env,key,greedy=greedy,training=training,deadline=budget.deadline)
    finally:del env.step


def training_collect(model,arm,env,key,reference,budget):
    """Save replay roots and continuation hashes, without retaining cloned envs."""
    before=torch.get_rng_state().clone()
    if arm=='standard':
        ep=plain_collect(model,env,key,budget,'actual',training=True,greedy=False)
        ep.update(reference_rewards=reference.copy(),roots=[],policy_rng_before=before,
                  actual_policy_rng=torch.get_rng_state().clone(),reference_days=0,skipped_identical=0)
        return ep
    hashes={};continuation=model.continuation
    def counted_continuation(snapshot,role,action,**kwargs):
        previous=dict(budget.context);first=snapshot.day
        budget.context.update(root_role=role,root_day=first);step=snapshot.step
        def counted(action):value=step(action);budget.done();return value
        snapshot.step=counted
        try:
            rr,trace=continuation(snapshot,role,action,**kwargs)
            hashes[role,first]=hash_value(trace);return rr,trace
        finally:del snapshot.step;budget.context=previous
    model.continuation=counted_continuation
    try:
        ep=model.collect(CountedEnvironment(env,budget),key,budget=budget,deadline=budget.deadline,keep_proof_snapshots=True)
    finally:del model.continuation
    roots=[]
    for role,row,snapshot,rng in ep.pop('proof_snapshots'):
        identity=np.array_equal(row['action'],row['reference_action'])
        trace_digest=hash_value(ep['trace'][row['start']:]) if identity else hashes[role,row['start']+1]
        roots.append(dict(role=role,root_rng=rng,reference_trace_sha256=trace_digest))
    ep.update(whole_reference_rewards=reference.copy(),roots=roots,policy_rng_before=before)
    assert torch.equal(torch.get_rng_state(),ep['actual_policy_rng'])
    return ep


def episode_metrics(ep,covers):
    from src.algos.parameter_policy_gradient import order_action
    trace=ep['trace'];assert len(trace)==28
    demand=np.asarray([r['info']['demand'] for r in trace]);sales=np.asarray([r['info']['sales'] for r in trace])
    changed=sum(int(np.count_nonzero(r['actions']!=order_action(np.asarray(r['obs']),r['day'],covers))) for r in trace)
    prices=np.concatenate([r['price_actions'] for r in trace if r['price_actions'] is not None])
    return dict(reward_day=float(np.mean(ep['rewards'])),episode_return=float(sum(ep['rewards'])),
        fill=float(sales.sum()/max(demand.sum(),1e-12)),fill_by_sku=(sales.sum(0)/np.maximum(demand.sum(0),1e-12)).tolist(),
        absolute_service=bool(sales.sum()/max(demand.sum(),1e-12)>=.95),
        order_changes=changed,price_changes=int(np.count_nonzero(prices!=4)),price_indices=np.unique(prices).tolist(),
        lost_units=float((demand-sales).sum()),stockout_sku_days=int(np.count_nonzero(demand>sales+1e-8)))


def analysis(records,*,replicates=20000,seed=20280108):
    """One-trained-seed paired SCENARIO uncertainty, six-effect fixed family."""
    keys=[r['key'] for r in records['anchor']]
    assert len(keys)==len(set(keys)) and len(keys)>1
    assert all([r['key'] for r in rows]==keys for rows in records.values())
    rng=np.random.default_rng(seed);draws=rng.integers(len(keys),size=(replicates,len(keys)))
    effects={};means={arm:{metric:float(np.mean([r[metric] for r in rows])) for metric in ('reward_day','fill')}
        for arm,rows in records.items()}
    for control in ('anchor','whole','standard'):
        contrast={}
        for metric in ('reward_day','fill'):
            delta=np.asarray([a[metric]-b[metric] for a,b in zip(records['coordinate'],records[control])])
            ci=np.quantile(delta[draws].mean(1),[.05/12,1-.05/12])
            contrast[metric]=dict(mean=float(delta.mean()),ci=ci.tolist())
        contrast['gate']=bool(contrast['reward_day']['ci'][0]>0 and contrast['fill']['ci'][0]>=-.005 and
                              (control!='anchor' or contrast['reward_day']['mean']>=.5))
        effects[control]=contrast
    return dict(means=means,effects=effects,all_pilot_gates=all(c['gate'] for c in effects.values()),
        confidence=1-.05/6,bootstrap_replicates=replicates,bootstrap_seed=seed,trained_seeds=1,
        uncertainty='paired scenario resampling conditional on one trained seed',full_goal_achieved=False)


def source_inventory():
    paths={p.as_posix() for d in ('src','scripts') for p in Path(d).rglob('*.py')}
    paths|={p.as_posix() for p in Path('outputs').glob('*conditional_credit*.py')}
    paths|={PROTOCOL,CALIBRATION,COVERS,'requirements.txt','Docs/1. Research Design and Locked Configuration',
        'src/config/skus.json','data/processed/aligned_skus.csv','data/processed/aligned_skus.key.json',
        'outputs/coordinate_reference_mappo.py','outputs/coordinate_credit_prototype.py','outputs/coordinate_reference_env_adapter.py',
        'outputs/linear_coordinate_diagnostic.py','outputs/credit_gradient_analytic.py',
        'outputs/direct_mappo/credit_signal_v1/manifest.json','outputs/direct_mappo/credit_signal_v1_audit/report.json',
        'outputs/direct_mappo/linear_conditional_credit_correctness_v1.xml','outputs/direct_mappo/linear_conditional_credit_proof_v1.json',
        'outputs/direct_mappo/conditional_credit_replay_correctness_v2.xml','outputs/direct_mappo/conditional_credit_replay_proof_v2.json'}
    diagnostic=json.loads(Path('outputs/direct_mappo/credit_signal_v1/manifest.json').read_text())
    assert all(digest(p)==h for p,h in diagnostic['hashes'].items())
    paths|=set(diagnostic['hashes'])
    assert json.loads(Path('outputs/direct_mappo/credit_signal_v1_audit/report.json').read_text())['status']=='PASS'
    from xml.etree import ElementTree as ET
    for name,count in (('linear_conditional_credit_correctness_v1',2),('conditional_credit_replay_correctness_v2',3)):
        tests=ET.parse(f'outputs/direct_mappo/{name}.xml').getroot()
        assert sum(int(s.attrib.get('tests',0)) for s in tests.iter('testsuite'))==count
        assert not list(tests.iter('failure')) and not list(tests.iter('error'))
    proof=json.loads(Path('outputs/direct_mappo/conditional_credit_replay_proof_v2.json').read_text())
    assert proof['status']=='PASS_SYNTHETIC' and proof['calibrated_days']==0 and proof['updates']==6
    assert all(digest(p)==h for p,h in proof['source_hashes'].items())
    return {p:digest(p) for p in sorted(paths)}
