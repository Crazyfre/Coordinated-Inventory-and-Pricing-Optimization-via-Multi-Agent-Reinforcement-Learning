"""Freeze CROSS_SKU_RECOVERY_V2_PROTOCOL.md once; no scientific execution."""
import json
from pathlib import Path
from scripts.recovery_campaign_plan import build
from scripts.recovery_invocation import validate_schedule
from scripts.recovery_overhead import validate
from scripts.recovery_source_inventory import inventory, runtime_identity, validate_coverage
from scripts.verify_recovery_science import check_plan
from src.experiments.execution_journal import digest, atomic_json


def main():
    repository = Path.cwd().resolve()
    root = repository/'outputs/structured_mappo/cross_sku_recovery_v2'
    destination = repository/'outputs/structured_mappo/cross_sku_recovery_v2_launch.json'
    if destination.exists() or root.exists() or root.with_name(root.name+'_invocations').exists():
        raise RuntimeError('fresh freeze only; existing declaration/archive must not be replaced')
    protocol = 'CROSS_SKU_RECOVERY_V2_PROTOCOL.md'
    paths = dict(calibration='outputs/calibration_correction/identifiable_v2/train_only_nb_identifiable_v2.json',
                 ordering='outputs/rl_discovery/parameter_confirmation_v1/sku_seed211_final.json')
    inputs = {key:dict(path=path, sha256=digest(path)) for key,path in paths.items()}
    hashes = inventory(repository, inputs, protocol)
    for name in ('CROSS_SKU_PRICING_PROTOCOL.md', 'CROSS_SKU_FEATURE_SPEC.md', 'CROSS_SKU_RECOVERY_PROTOCOL.md', 'outputs/recovery_real_path_v1/declaration.json', 'outputs/recovery_real_path_v1/report.json'):
        hashes[name] = digest(name)
    study = dict(namespaces=dict(scan=2770000, future=2780000, assessment=2790000, bootstrap=20261128),
        seconds=dict(scan=60., selection=60., verify=60., group=60., fit=300.,
                     assessment=60., replays=60., summary=60.),
        caps=dict(days=501872, fit=14400., assessment=1200., total=15600.),
        source_hashes=hashes, inputs=inputs,
        callback=dict(module='scripts.recovery_campaign_dispatch', function='dispatch'))
    plan = build(study)
    schedule = [[0,129]] + [[129+272*i,129+272*(i+1)] for i in range(8)]
    schedule += [[2305,2306]] + [[2306+32*i,2306+32*(i+1)] for i in range(4)] + [[2434,2436]]
    overhead = dict(per_invocation_seconds=10., maximum_invocations=22, fit=100., assessment=60., total=160.)
    declaration = dict(study=study, protocol=protocol, runtime=runtime_identity(), schedule=schedule, overhead=overhead)
    validate(overhead)
    validate_schedule(plan, schedule)
    check_plan(study, plan)
    validate_coverage(repository, study, protocol)
    packet = dict(root=str(root), declaration=declaration, plan=plan)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, packet)
    print(json.dumps(dict(packet=str(destination), sha256=digest(destination),
        units=len(plan['units']), invocations=len(schedule), maximum_successful_days=sum(u['days'] for u in plan['units']),
        source_files=len(hashes), new_simulator_days=0), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()

