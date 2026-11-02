"""Science components for future recovery campaign; no campaign entry point."""
import numpy as np
from src.algos.cross_sku_pricing_features import features as pressure
from src.algos.cross_sku_rollout import matrix, fit
from scripts.screen_cross_sku_pricing import reset, collect
from scripts.diagnose_conditional_pricing import advance, suffix
from scripts.diagnose_counterfactual_rollouts import clone_environment
from scripts.audit_repeated_futures import snapshot, resample_future
from scripts.recovery_scientific_units import paired_group


class CountedEnvironment:
    """Intercept every step in legacy read-only helpers, including clones."""
    def __init__(self, env, budget):
        self.env, self.budget = env, budget

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        self.budget.before_step()
        result = self.env.step(action)
        self.budget.check()
        return result


def select(observations, episode, base, namespace):
    if set(observations) != {1,8,15,22}:
        raise ValueError('all weekly observations required')
    candidates = [(d,i) for d in (1,8,15,22) for i in range(10)]
    scores = np.concatenate([pressure(observations[d], base)[:,3] for d in (1,8,15,22)])
    best = int(np.argmax(scores))
    remaining = [i for i in range(40) if i != best]
    rng = np.random.default_rng(np.random.SeedSequence([namespace,episode,991]))
    return [list(candidates[best]), list(candidates[remaining[int(rng.integers(39))]])]


def keys(namespace, ep, day, sku, replica):
    return [int(np.random.SeedSequence([namespace,ep,day,sku,replica,z]).generate_state(1,dtype=np.uint64)[0]) for z in (0,1)]


def scan(env, covers, ep, namespaces, budget):
    key = namespaces['scan']+ep
    reset(env, key)
    trace = suffix(CountedEnvironment(env,budget), covers, 0, 4, {'days':0}, budget.deadline)
    observations = {r['day']:r['pricing_obs'] for r in trace if r['pricing_obs'] is not None}
    return dict(episode=ep,key=key,trace=trace,observations=observations,
                selected=select(observations,ep,env.base_price,namespaces['scan']))


def freeze(scans, base, namespaces):
    if len(scans) != 128 or [r['episode'] for r in scans] != list(range(128)):
        raise ValueError('complete ordered scan grid required')
    selections = []
    for record in scans:
        obs = {int(k):v for k,v in record['observations'].items()}
        actual = {r['day']:r['pricing_obs'] for r in record['trace'] if r['pricing_obs'] is not None}
        if (record['key'] != namespaces['scan']+record['episode'] or len(record['trace']) != 28 or
                obs != actual or record['selected'] != select(obs,record['episode'],base,namespaces['scan'])):
            raise ValueError('invalid scan selection or identity')
        selections.append({k:record[k] for k in ('episode','key','observations','selected')})
    length = sum(29-d for r in selections for d,_ in r['selected'])
    return dict(selections=selections,suffix_length_sum=length,successful_days=78960+59*length)


def verify_window(env, covers, original, selection, namespaces, budget):
    ep = original['episode']
    if original['key'] != selection['key'] or selection['episode'] != ep:
        raise ValueError('selection/window identity mismatch')
    reset(env, original['key'])
    counted = CountedEnvironment(env,budget)
    trace, states = [], []
    for day in range(1,29):
        for rank,(d,sku) in enumerate(selection['selected']):
            if day != d:
                continue
            before = snapshot(env)
            obs = env.pricing_observation().tolist()
            expected = selection['observations'].get(str(day),selection['observations'].get(day))
            if obs != expected:
                raise ValueError('weekly observation differs from frozen scan')
            probe = suffix(CountedEnvironment(clone_environment(env),budget), covers, sku, 4, {'days':0}, budget.deadline)
            if probe != original['trace'][day-1:]:
                raise ValueError('original suffix replay mismatch')
            future_keys = keys(namespaces['future'],ep,day,sku,0)
            replays = [suffix(CountedEnvironment(resample_future(env,future_keys),budget), covers, sku, a,
                              {'days':0}, budget.deadline) for a in (0,4)]
            if snapshot(env) != before:
                raise ValueError('verification mutated baseline')
            states.append(dict(rank=rank,day=day,sku=sku,before=before,obs=obs,probe=probe,
                               keys=future_keys,replays=replays))
        row,_ = advance(counted,covers,day,0,4,{'days':0},budget.deadline)
        trace.append(row)
    if trace != original['trace'] or len(states) != 2:
        raise ValueError('original window replay mismatch')
    return dict(episode=ep,key=original['key'],original_replay=trace,states=sorted(states,key=lambda s:s['rank']))


def group(env, covers, original, verification, rank, group_index, namespaces, budget):
    ep = original['episode']
    state = verification['states'][rank]
    day,sku = state['day'],state['sku']
    futures = [dict(replica=r,keys=keys(namespaces['future'],ep,day,sku,r))
               for r in range(4*group_index,4*group_index+4)]
    result = paired_group(env,covers,original['key'],day,sku,futures,budget,state['before'])
    if result['prefix'] != original['trace'][:day-1]:
        raise ValueError('group prefix differs from original')
    if group_index == 0 and [b['trace'] for b in result['draws'][0]['branches']] != state['replays']:
        raise ValueError('replica0 differs from independent replay')
    result.update(episode=ep,rank=rank,group=group_index)
    return result


def fit_samples(samples, base):
    if [(s['episode'],s['rank']) for s in samples] != [(e,r) for e in range(128) for r in range(2)]:
        raise ValueError('complete ordered256 samples required')
    if any(len(s['gains']) != 32 for s in samples):
        raise ValueError('all32 paired futures required')
    targets = np.asarray([np.mean(s['gains'])/base[s['sku']] for s in samples])
    matrices = {arm:np.array([matrix(s['obs'],s['day'],base,arm)[s['sku']] for s in samples]) for arm in ('local','cross')}
    return dict(samples=samples,base_prices=np.asarray(base).tolist(),targets=targets.tolist(),
                matrices={k:v.tolist() for k,v in matrices.items()},
                models={arm:fit(x,targets,arm) for arm,x in matrices.items()})


def assemble_sample(verification, rank, groups):
    state = verification['states'][rank]
    ep = verification['episode']
    if len(groups) != 8 or [g['group'] for g in groups] != list(range(8)):
        raise ValueError('eight ordered groups required')
    gains = []
    for index, group_record in enumerate(groups):
        if (group_record['episode'] != ep or group_record['rank'] != rank or
                group_record['day'] != state['day'] or group_record['sku'] != state['sku'] or
                group_record['before'] != state['before'] or
                [d['replica'] for d in group_record['draws']] != list(range(4*index,4*index+4))):
            raise ValueError('group identity/state/replica mismatch')
        for draw in group_record['draws']:
            if [b['action'] for b in draw['branches']] != [0,4]:
                raise ValueError('paired action0/action4 required')
            values = []
            for branch in draw['branches']:
                trace = branch['trace']
                if [r['day'] for r in trace] != list(range(state['day'],29)):
                    raise ValueError('complete suffix required')
                value = sum(.99**i*r['reward'] for i,r in enumerate(trace))
                if not np.isfinite(value) or value != branch['value']:
                    raise ValueError('branch return mismatch')
                values.append(value)
            gains.append(values[0]-values[1])
    return dict(episode=ep,rank=rank,day=state['day'],sku=state['sku'],obs=state['obs'],gains=gains)


def analyze(units, namespaces):
    if len(units) != 128 or [u['episode'] for u in units] != list(range(128)):
        raise ValueError('complete ordered assessment grid required')
    labels = ('cross','local','baseline','heuristic')
    rows = [r['result'] for u in units for r in u['records']]
    ix = {(r['label'],r['key']):r for r in rows}
    assessment_keys = list(range(namespaces['assessment'],namespaces['assessment']+128))
    if len(ix) != len(rows) or set(ix) != {(l,k) for l in labels for k in assessment_keys}:
        raise ValueError('unique complete controller/key grid required')
    draws = np.random.default_rng(namespaces['bootstrap']).integers(128,size=(10000,128))
    effects = {}
    for reference in ('baseline','local','heuristic'):
        for metric in ('reward_day','fill'):
            values = np.array([ix['cross',k][metric]-ix[reference,k][metric] for k in assessment_keys])
            if not np.isfinite(values).all():
                raise ValueError('nonfinite assessment metric')
            effects[f'{reference}:{metric}'] = dict(mean=float(values.mean()),
                ci991667=np.quantile(values[draws].mean(1),[.05/12,1-.05/12]).tolist())
    passes = {r:effects[f'{r}:reward_day']['ci991667'][0]>0 and
              effects[f'{r}:fill']['ci991667'][0]>=-.005 for r in ('baseline','local','heuristic')}
    return dict(effects=effects,primary_pass=bool(passes['baseline'] and effects['baseline:reward_day']['mean']>=.5),
                representation_pass=bool(passes['local']),heuristic_pass=bool(passes['heuristic']),
                means={l:{m:float(np.mean([ix[l,k][m] for k in assessment_keys]))
                          for m in ('reward_day','fill')} for l in labels},
                price_counts={l:np.sum([ix[l,k]['price_counts'] for k in assessment_keys],axis=0).tolist() for l in labels})


def assessment(env, covers, ep, models, namespaces, budget):
    records = []
    key = namespaces['assessment']+ep
    for label in ('cross','local','baseline','heuristic'):
        result,trace = collect(CountedEnvironment(env,budget),covers,key,label,models,{'days':0},budget.deadline)
        records.append(dict(label=label,key=key,result=result,trace=trace))
    return dict(episode=ep,records=records)
