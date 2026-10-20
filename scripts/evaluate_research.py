"""Reproducible paired matrix. Invoke with python -m scripts.evaluate_research."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from src.experiments.research import (make_research_env, SCENARIOS, ABLATIONS,
    training_statistics_view, fit_training_calibration)
from src.experiments.phase4 import heuristic
from src.demand.alignment import build_aligned_dataset
from src.demand.identifiable_calibration import validate_research_training_calibration
from src.algos.pricing_schema import pricing_schema
from src.eval.research import seed_library, evaluate_episode, summarize, paired_difference, validate_final


def policies(method, env, checkpoint=None):
    from src.baselines.retail import EOQPolicy, MyopicPricingPolicy
    order = lambda e, o: heuristic(e)
    price = lambda o: 2
    if method == 'B1':
        return order, price
    if method == 'B2':
        return EOQPolicy(), price
    if method == 'B3':
        return order, MyopicPricingPolicy(env)
    if method == 'B6':
        from src.algos.mappo import HeterogeneousMAPPO
        model = HeterogeneousMAPPO(env, hidden=checkpoint.get('hidden',64),
                                   normalize_inputs=checkpoint.get('stable',False),
                                   target_kl=.02 if checkpoint.get('stable',False) else None)
        model.load(checkpoint['path'])
        return (lambda e,o: model.replenishment.act(o,greedy=True)[0],
                model.pricing_action)
    if method in ('B4','B5','B7'):
        if method == 'B4':
            from src.algos.independent_ppo import IndependentPPO as Model
        elif method == 'B5':
            from src.algos.joint_ppo import JointPPO as Model
        else:
            if checkpoint.get('variant_family') not in ('B4','B5','B6') or not checkpoint.get('variant_name'):
                raise ValueError('B7 not selected/frozen; do not invent proposed variant')
            return policies(checkpoint['variant_family'],env,checkpoint)
        model = Model(env, **checkpoint.get('model_kwargs',{}))
        model.load(checkpoint['path'])
        return model.order_action, model.pricing_action
    raise ValueError('unknown method')


def _csv(path, rows):
    if not rows:
        path.write_text('',encoding='utf-8'); return
    with path.open('w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def select_checkpoints(checkpoints, training_scenario='S2'):
    selected=[c for c in checkpoints if c.get('training_scenario','legacy') in ('legacy',training_scenario)]
    keys=[(c['method'],c.get('ablation','none'),c.get('training_seed','unknown')) for c in selected]
    if len(keys)!=len(set(keys)):
        raise ValueError('duplicate method/ablation/training seed checkpoints; select one training regime')
    return selected


def checkpoint_pricing_scope(checkpoint, *, requested=None, final=False):
    """Missing/old descriptive config scopes identify historical shared actors."""
    legacy_description = "one common scalar multiplier for all SKUs (existing joint env restriction)"
    raw = checkpoint.get("pricing_scope")
    scope = "shared" if raw in (None, legacy_description) else raw
    if scope not in ("shared", "per_sku"):
        raise ValueError("unknown checkpoint pricing scope")
    if requested is not None and scope != requested:
        raise ValueError("checkpoint pricing scope disagrees with evaluation scope")
    if final and (raw not in ("shared", "per_sku") or not checkpoint.get("action_schema")):
        raise ValueError("final checkpoint requires explicit pricing action schema")
    return scope


def run(cfg):
    import torch
    torch.set_num_threads(1)
    output = Path(cfg['output']); output.mkdir(parents=True,exist_ok=True)
    if (output/'episodes.json').exists():
        raise ValueError('output already contains a run; choose a fresh directory')
    final = cfg.get('mode')=='final'
    split = cfg.get('split','validation')
    if split=='test' and not final:
        raise ValueError('test outcomes forbidden in development mode')
    if final and split!='test':
        raise ValueError('final requires held-out test split')
    if final and set(cfg.get('methods',[])) != {f'B{i}' for i in range(1,8)}:
        raise ValueError('final requires B1-B7 methods')
    if final and set(cfg.get('elasticity_sweep',[])) != {-.5,-1.5,-3.}:
        raise ValueError('final requires the complete controlled elasticity sweep')
    requested_scope = cfg.get('pricing_scope')
    if requested_scope is not None and requested_scope not in ("shared", "per_sku"):
        raise ValueError("evaluation pricing_scope must be shared or per_sku")
    if final and requested_scope != "per_sku":
        raise ValueError("final design evaluation requires explicit per_sku pricing_scope")
    episodes = cfg.get('episodes',5)
    data = build_aligned_dataset()
    aligned = training_statistics_view(data)
    if cfg.get('calibration'):
        params = json.loads(Path(cfg['calibration']).read_text())
        validate_research_training_calibration(params)
    else:
        params = None  # legacy, explicitly development-only
    checkpoints = cfg.get('checkpoints',[]) + (discover_checkpoints(cfg['checkpoint_root']) if cfg.get('checkpoint_root') else [])
    checkpoints = select_checkpoints(checkpoints,cfg.get('training_scenario','S2'))
    if any(c.get('ablation')=='A1' and c['method']!='B4' for c in checkpoints):
        raise ValueError('A1 requires an independent local-critic B4 checkpoint')
    if final:
        for checkpoint in checkpoints:
            checkpoint_pricing_scope(checkpoint, requested=requested_scope, final=True)
        if params is None:
            raise ValueError('final requires train-only calibration')
        for method in ('B4','B5','B6','B7'):
            group = [c for c in checkpoints if c['method']==method and c.get('ablation','none')=='none']
            validate_final(params,group,episodes)
        for ablation in ABLATIONS:
            group = [c for c in checkpoints if c.get('ablation')==ablation]
            validate_final(params,group,episodes)
        if set(cfg.get('scenarios',[])) != set(SCENARIOS) or set(cfg.get('ablations',[])) != set(ABLATIONS):
            raise ValueError('final requires full S1-S8/A1-A7 coverage')
    seeds = seed_library(episodes,cfg.get('seed',20261012))
    (output/'seeds.json').write_text(json.dumps(seeds,indent=2))
    snapshot = dict(cfg,checkpoints=checkpoints)
    snapshot.pop('checkpoint_root',None)
    (output/'run_config.json').write_text(json.dumps(snapshot,indent=2))
    rows, coverage, metrics, comparisons = [], [], [], []
    conditions = [(s,'none') for s in cfg.get('scenarios',list(SCENARIOS))]
    if 'S4' in cfg.get('scenarios',list(SCENARIOS)):
        conditions += [(f'S4_e{e:g}','none') for e in cfg.get('elasticity_sweep',[]) if e!=-3.]
    conditions += [('S2',a) for a in cfg.get('ablations',list(ABLATIONS))]
    methods = cfg.get('methods',['B1','B2','B3','B4','B5','B6','B7'])
    for scenario, ablation in conditions:
        for method in methods:
            matching = [c for c in checkpoints if c['method']==method and c.get('ablation','none')==ablation]
            probe = False
            if ablation!='none' and not matching and method=='B6' and ablation!='A1' and not final:
                matching = [c for c in checkpoints if c['method']==method and c.get('ablation','none')=='none']
                probe = True
            if ablation!='none' and method in ('B1','B2','B3'):
                continue  # component ablations target learned architectures
            if method not in ('B1','B2','B3') and not matching:
                coverage.append(dict(scenario=scenario,ablation=ablation,method=method,status='missing_checkpoint'))
                continue
            models = matching or [None]
            for model in models:
                elasticity = float(scenario[4:]) if scenario.startswith('S4_e') else None
                scope = (checkpoint_pricing_scope(model, requested=requested_scope, final=final)
                         if model else (requested_scope or 'shared'))
                env = make_research_env('S4' if elasticity is not None else scenario, None if ablation=='none' else ablation,
                    split=split, days=cfg.get('days',28), params=params,aligned=aligned,elasticity=elasticity,pricing_scope=scope)
                if model and model.get("action_schema") is not None and model["action_schema"] != pricing_schema(env):
                    raise ValueError("manifest pricing action schema disagrees with reconstructed environment")
                order, price = policies(method,env,model)
                group = []
                for seed in seeds:
                    r = evaluate_episode(env,order,price,seed)
                    r.update(scenario=scenario,ablation=ablation,method=method,
                        training_seed=model.get('training_seed','unknown') if model else 'not_applicable',
                        pricing_scope=scope,kind='inference_probe' if probe else ('checkpoint_evaluation' if model else 'baseline'))
                    rows.append(r); group.append({k:v for k,v in r.items() if isinstance(v,(int,float)) and k!='training_seed'})
                label = str(model.get('training_seed','unknown')) if model else 'not_applicable'
                for metric, stats in summarize(group).items():
                    metrics.append(dict(scenario=scenario,ablation=ablation,method=method,training_seed=label,
                                        metric=metric,kind=rows[-1]['kind'],**stats))
                coverage.append(dict(scenario=scenario,ablation=ablation,method=method,
                    status='inference_probe_not_retrained' if probe else 'evaluated',training_seed=label,
                    episodes=episodes,checkpoint=model['path'] if model else 'none'))
                (output/'episodes.json').write_text(json.dumps(rows,indent=2))
    # Exact seed matching; compare each checkpoint against paired B1 environment.
    for scenario in sorted({s for s,a in conditions if a=='none'}):
        baseline = {r['seed']:r for r in rows if r['scenario']==scenario and r['ablation']=='none' and r['method']=='B1'}
        groups = {}
        for r in rows:
            if r['scenario']==scenario and r['ablation']=='none' and r['method']!='B1':
                groups.setdefault((r['method'],str(r['training_seed'])),[]).append(r)
        for (method,trainseed), group in groups.items():
            if baseline and episodes>=2:
                for metric in ('profit','fill_rate','stockout_rate'):
                    d = paired_difference([r[metric] for r in group],[baseline[r['seed']][metric] for r in group])
                    comparisons.append(dict(scenario=scenario,method=method,training_seed=trainseed,
                                            reference='B1',metric=metric,**d))
    _csv(output/'metrics.csv',metrics); _csv(output/'paired_comparisons.csv',comparisons)
    _csv(output/'episodes.csv',rows)
    (output/'coverage.json').write_text(json.dumps(coverage,indent=2))
    hashes = {c['path']:hashlib.sha256(Path(c['path']).read_bytes()).hexdigest() for c in checkpoints}
    result = dict(episode_rows=len(rows),evaluated_conditions=sum('episodes' in c for c in coverage),
                  held_out_test_used=split=='test',mode=cfg.get('mode','development'),
                  calibration='train_only' if params else 'legacy_fit_through_1913',
                  checkpoint_sha256=hashes,training_seed_counts={m:len({c.get('training_seed','unknown') for c in checkpoints if c['method']==m}) for m in methods},
                  complete_final_scale=False if not final else True)
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    return result


def discover_checkpoints(root):
    """Only completed sibling runs; never manufacture training provenance."""
    mapping={'independent':'B4','joint':'B5','mappo':'B6'}
    found=[]
    for path in sorted(Path(root).rglob('config.json')):
        parent=path.parent
        if not (parent/'summary.json').exists() or not (parent/'checkpoint.pt').exists():
            continue
        config=json.loads(path.read_text())
        summary=json.loads((parent/'summary.json').read_text())
        if config.get('method') not in mapping or not summary.get('checkpoint_roundtrip_verified'):
            continue
        clean = config.get('random_train_window_per_episode',False) and not config.get('held_out_test_used',True)
        entry=dict(method=mapping[config['method']],path=str(parent/'checkpoint.pt'),
            training_seed=config['seed'],stable=config.get('stable',True),updates=config['updates'],
            training_days=config['days'],ablation='none',frozen=False,
            training_scenario=config.get('scenario','legacy'), source_config=str(path),
            pricing_scope=checkpoint_pricing_scope(config),
            action_schema=config.get('action_schema'),
            provenance='clean train-only development; NOT frozen/final' if clean else 'sibling development regime; legacy calibration; NOT final-eligible')
        if clean:
            entry.update(training_end_day=config['train_window_range'][1]+config['days']-1,
                         split='train',calibration_sha256=config.get('calibration_sha256'))
        found.append(entry)
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--prepare-calibration',action='store_true')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.prepare_calibration:
        path = Path(cfg['calibration']); path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            raise ValueError('calibration already exists; do not silently replace')
        path.write_text(json.dumps(fit_training_calibration(),indent=2)); print(path)
    else:
        print(json.dumps(run(cfg),indent=2))


if __name__=='__main__':
    main()
