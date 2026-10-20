"""Export all five verified mean policies without selecting a favourable seed."""
import argparse
import json
from pathlib import Path
from scripts.train_compact_residual import digest, write_json, INPUTS
from src.algos.learned_cover_policy import LearnedCoverPolicy


def export(study):
    study = Path(study)
    summary = json.loads((study/'summary.json').read_text())
    verification = json.loads((study/'verification.json').read_text())
    if not summary['analysis']['learning_gate'] or verification['status'] != 'passed' or not verification['confirmed_gate']:
        raise ValueError('passing independently verified study required')
    manifest = json.loads((study/'manifest.json').read_text())
    if manifest['arm'] != 'sku':
        raise ValueError('export is specific to the ten-cover selected method')
    calibration = json.loads(Path(INPUTS['calibration']).read_text())
    order = calibration['series_order']
    frozen = json.loads((study/'frozen_checkpoints.json').read_text())
    output = study/'policies'; output.mkdir(exist_ok=False)
    files = []
    for seed in manifest['seeds']:
        checkpoint = study/f'sku_seed{seed}_final.json'
        if digest(checkpoint) != frozen[str(seed)]:
            raise ValueError('checkpoint changed after assessment')
        state = json.loads(checkpoint.read_text())
        assert state['steps'] == 64 and state['arm'] == 'sku'
        LearnedCoverPolicy(state['mean'], order)
        path = output/f'seed{seed}.json'
        write_json(path, dict(variant=LearnedCoverPolicy.variant, training_seed=seed,
              execution='deterministic_mean', cover_means=state['mean'], series_order=order,
              calibration_sha256=digest(INPUTS['calibration']), source_checkpoint_sha256=digest(checkpoint),
              summary_sha256=digest(study/'summary.json'), verification_sha256=digest(study/'verification.json'),
              conditions=dict(scenario='S2', price_multiplier=1.2, pricing_scope='shared_fixed', episode_days=28,
                              order_grid=list(range(0, 201, 5)), capacity=500, billing='accepted_delivered_units'),
              limits='Ordering-only research controller; no test-period or coordination claim; not a demo promotion.'))
        loaded = LearnedCoverPolicy.load(path, series_order=order, calibration_path=INPUTS['calibration'])
        assert loaded.parameters.tolist() == state['mean']
        files.append(dict(seed=seed, path=path.name, sha256=digest(path)))
    write_json(output/'manifest.json', dict(selection='all five seeds; no best-seed selection', files=files,
               loader='src.algos.learned_cover_policy.LearnedCoverPolicy', loader_sha256=digest('src/algos/learned_cover_policy.py')))
    print('Exported all five verified deterministic mean controllers; no model replacement.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--study', required=True); export(p.parse_args().study)
