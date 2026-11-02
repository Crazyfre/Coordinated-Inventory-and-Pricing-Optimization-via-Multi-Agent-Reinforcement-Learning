"""Bound scientific callback for future recovery campaign; no launch command.

Reads only dependencies named by the immutable plan, validates their hashes and
input identities, and retains one state's eight groups while assembling fits.
"""
import json
from pathlib import Path
import numpy as np
import torch
from src.experiments.execution_journal import digest
from src.experiments.research import make_research_env
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from scripts.recovery_campaign_plan import build, bind_request, identity
from scripts import recovery_campaign_science as science


def validate_request(request):
    study, unit, records = (request[k] for k in ('study','unit','dependencies'))
    if study['callback'] != dict(module=__name__,function='dispatch'):
        raise ValueError('callback differs from frozen dispatcher')
    bind_request(study,unit,records)
    for path, checksum in study['source_hashes'].items():
        if digest(path) != checksum:
            raise ValueError(f'frozen source changed: {path}')
    expected = {u['item']:u for u in build(study)['units']}
    for record in records:
        if record['input_identity'] != expected[record['item']]['input_identity']:
            raise ValueError('dependency input identity mismatch')
    return study,unit,{r['item']:r for r in records}


def load_dependency(record):
    path = Path(record['path'])
    if digest(path) != record['sha256']:
        raise ValueError('dependency artifact hash mismatch')
    artifact = json.loads(path.read_text(encoding='utf-8'))
    if artifact['item'] != record['item'] or artifact['input_identity'] != record['input_identity']:
        raise ValueError('dependency artifact identity mismatch')
    return artifact['payload']


def environment(study,context):
    for name in ('ordering','calibration'):
        record = study['inputs'][name]
        if digest(record['path']) != record['sha256']:
            raise ValueError(f'changed {name} input')
    signature = identity(study['inputs'])
    if context.get('campaign_inputs') != signature:
        torch.set_num_threads(1)
        calibration = load_identifiable_training_calibration(study['inputs']['calibration']['path'])
        covers = np.array(json.loads(Path(study['inputs']['ordering']['path']).read_text())['mean'])
        env = make_research_env('S2',split='train',start=15,params=calibration,pricing_scope='per_sku')
        context.update(campaign_inputs=signature,campaign_env=env,campaign_covers=covers)
    return context['campaign_env'],context['campaign_covers']


def dispatch(request,budget,context):
    study,unit,records = validate_request(request)
    budget.check()
    spec = unit['specification']
    ns = study['namespaces']
    load = lambda item:load_dependency(records[item])
    env,covers = environment(study,context)
    started_days = budget.days
    kind = spec['kind']
    if kind == 'scan':
        result = science.scan(env,covers,spec['episode'],ns,budget)
        days = 28
    elif kind == 'selection':
        result = science.freeze([load(f'scan_{e:03d}') for e in range(128)],env.base_price,ns)
        days = 0
    elif kind in ('verify','group'):
        ep = spec['episode']
        frozen = load('selection')
        original = load(f'scan_{ep:03d}')
        selected = frozen['selections'][ep]
        if kind == 'verify':
            result = science.verify_window(env,covers,original,selected,ns,budget)
            days = 28+3*sum(29-d for d,_ in selected['selected'])
        else:
            verification = load(f'verify_{ep:03d}')
            result = science.group(env,covers,original,verification,spec['rank'],spec['group'],ns,budget)
            day = result['day']
            days = day-1+8*(29-day)
    elif kind == 'fit':
        frozen = load('selection')
        samples = []
        for ep in range(128):
            verification = load(f'verify_{ep:03d}')
            for rank in range(2):
                groups = [load(f'group_{ep:03d}_{rank}_{g}') for g in range(8)]
                sample = science.assemble_sample(verification,rank,groups)
                if [sample['day'],sample['sku']] != frozen['selections'][ep]['selected'][rank]:
                    raise ValueError('sample differs from frozen selection')
                samples.append(sample)
        result = science.fit_samples(samples,env.base_price)
        days = 0
    elif kind in ('assessment','replays'):
        models = load('fit')['models']
        result = science.assessment(env,covers,spec.get('episode',0),models,ns,budget)
        if kind == 'replays' and result != load('assess_000'):
            raise ValueError('assessment replay mismatch')
        days = 112
    elif kind == 'summary':
        units = [load(f'assess_{e:03d}') for e in range(128)]
        if load('replays') != units[0]:
            raise ValueError('replay differs from first assessment')
        result = science.analyze(units,ns)
        days = 0
    else:
        raise ValueError('unknown scientific unit')
    if budget.days-started_days != days:
        raise ValueError('scientific unit step count mismatch')
    budget.check()
    return result
