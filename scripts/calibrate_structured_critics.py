"""Predeclared value-only screen; frozen initial actors and separate scenarios."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.algos.structured_pricing_ppo import StructuredPricingPPO
from src.algos.structured_ordering_ppo import StructuredOrderingPPO
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env
from src.eval.critic_calibration import episode_data, fit, predictions, metrics
from scripts.pilot_structured_pricing_ppo import collect as collect_pricing
from scripts.pilot_structured_ordering_ppo import collect as collect_ordering
from scripts.train_clean_research import TrainingWindows
from scripts.train_compact_residual import INPUTS, digest, write_json


def run(output):
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    roots = {k: Path(f'outputs/structured_mappo/{k}_ppo_pilot_v1') for k in ['pricing', 'ordering']}
    sources = {str(p) for folder in ['src', 'scripts'] for p in Path(folder).rglob('*.py')}
    sources |= {INPUTS['calibration'], 'STRUCTURED_CRITIC_CALIBRATION_PROTOCOL.md'}
    for root in roots.values():
        assert json.loads((root / 'verification.json').read_text())['status'] == 'passed'
        sources |= {str(root / p) for p in ['manifest.json', 'initial.pt', 'verification.json']}
    hashes = {p: digest(p) for p in sorted(sources)}
    write_json(out / 'manifest.json', dict(hashes=hashes, total_episodes=386, maximum_seconds=900, test_scoring=False))
    torch.set_num_threads(1); started = time.perf_counter(); count = 0
    try:
        calibration = load_identifiable_training_calibration(INPUTS['calibration'])
        env = TrainingWindows(make_research_env('S2', split='train', start=15, days=28,
            params=calibration, pricing_scope='per_sku'), 28)
        env.reset(seed=0); results = {}
        for kind, first, second in [('pricing', 2270000, 2290000), ('ordering', 2280000, 2300000)]:
            root = roots[kind]; prior = json.loads((root / 'manifest.json').read_text())
            covers = np.asarray(prior['covers'])
            model = (StructuredPricingPPO(prior['state_dim'], 601) if kind == 'pricing' else
                     StructuredOrderingPPO(prior['state_dim'], covers, 701))
            saved = torch.load(root / 'initial.pt', weights_only=True, map_location='cpu')
            model.actor.load_state_dict(saved['actor']); model.critic.load_state_dict(saved['critic'])
            collector = collect_pricing if kind == 'pricing' else collect_ordering
            original = copy.deepcopy(model.critic); learned = copy.deepcopy(model.critic)
            fit_data = []
            with (out / f'{kind}_fit.jsonl').open('w') as handle:
                for key in range(first, first + 128):
                    torch.manual_seed(key + 400000)
                    data, _, _ = collector(env, model, covers, key, kind, handle, started + 900)
                    fit_data.append(episode_data(data, kind)); count += 1
            write_json(out / f'{kind}_fit_data.json', fit_data)
            norm, history = fit(learned, fit_data)
            if time.perf_counter() > started + 900: raise TimeoutError('critic screen time cap')
            torch.save(history, out / f'{kind}_history.pt')
            torch.save(learned.state_dict(), out / f'{kind}_critic.pt')
            write_json(out / f'{kind}_normalization.json', norm)
            frozen = {p: digest(out / p) for p in [f'{kind}_critic.pt', f'{kind}_normalization.json']}
            write_json(out / f'{kind}_assessment_freeze.json', frozen)
            assessment = []; target = None
            with (out / f'{kind}_assessment.jsonl').open('w') as handle:
                for key in range(second, second + 64):
                    torch.manual_seed(key + 400000)
                    data, _, trace = collector(env, model, covers, key, kind, handle, started + 900)
                    assessment.append(episode_data(data, kind)); count += 1
                    if key == second: target = trace
            with (out / f'{kind}_replay.jsonl').open('w') as handle:
                torch.manual_seed(second + 400000)
                _, _, trace = collector(env, model, covers, second, kind, handle, started + 900)
                count += 1; assert trace == target
            write_json(out / f'{kind}_assessment_data.json', assessment)
            pred = predictions(learned, original, norm, assessment)
            write_json(out / f'{kind}_predictions.json', pred)
            results[kind] = metrics(assessment, pred)
            assert all(digest(out / p) == h for p, h in frozen.items())
            assert all(torch.equal(v, saved['actor'][k]) for k, v in model.actor.state_dict().items())
            print(f'{kind}: ' + json.dumps(results[kind]), flush=True)
        assert count == 386 and all(digest(p) == h for p, h in hashes.items())
        elapsed = time.perf_counter() - started; assert elapsed <= 900
        write_json(out / 'summary.json', dict(status='complete', results=results, episodes=count,
            days=count * 28, seconds=elapsed, test_scoring=False,
            archive_hashes={p.name: digest(p) for p in out.iterdir() if p.is_file()}))
    except Exception as exc:
        write_json(out / 'failed_status.json', dict(error=str(exc), complete_episodes=count)); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); run(p.parse_args().output)
