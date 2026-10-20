"""Archive-only reconstruction of compact smoke policy choices and PPO updates."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.algos.compact_residual import CompactResidualPPO
from src.algos.structured_imitation import InventoryPositionLinearPolicy
from src.algos.imitation_ordering import teacher_indices
from scripts.train_compact_residual import digest, write_json


def verify(output):
    out = Path(output)
    torch.set_num_threads(1)
    manifest = json.loads((out/'manifest.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    for path, expected in manifest['hashes'].items():
        assert digest(path) == expected, path
    assert digest(out/'traces.jsonl') == summary['trace_sha256']
    for name, expected in summary['checkpoint_hashes'].items():
        assert digest(out/name) == expected
    rows = [json.loads(line) for line in (out/'traces.jsonl').read_text().splitlines()]
    assert len(rows) == 448 == summary['simulated_days']
    assert summary['episodes'] == 16 and not summary['campaign_executed']
    initial = torch.load(out/'initial.pt', weights_only=True)
    spec = initial['spec']
    model = CompactResidualPPO(spec['state_dim'], spec['coefficients'], spec['identity'], spec['provenance'], 44)
    model.load(out/'initial.pt')
    assert torch.count_nonzero(model.actor.weight) == 0
    episodes = [rows[i:i+28] for i in range(0, 448, 28)]
    for update in range(2):
        chunk = episodes[8*update:8*update+8]
        for ep_index, ep in enumerate(chunk):
            key = 1003000+8*update+ep_index
            rng = np.random.default_rng(np.random.SeedSequence([key, 314159]))
            expected_start = int(rng.integers(15, 1859))
            for day, row in enumerate(ep, 1):
                assert row['scenario'] == key and row['day'] == day and row['start'] == expected_start
                assert row['update'] == update
                x = np.asarray(row['obs'], np.float32)
                center = InventoryPositionLinearPolicy(10, 7, spec['coefficients']).predict(x)
                target = teacher_indices(torch.tensor(x), 10, 7, cover=7.5).numpy()
                default = np.clip(target, np.maximum(center-2, 0), np.minimum(center+2, 40))
                np.testing.assert_array_equal(default, row['default'])
                action, lp = model.actor.act(x, day, generator=model.generator)
                np.testing.assert_array_equal(action, row['action'])
                assert lp == row['log_prob']
                assert model.value(row['state']) == row['value']
                # Independent feature and distribution reconstruction, not actor.forward.
                stock, pipe, forecast = x[:10].astype(float), x[10:80].reshape(10, 7).astype(float), x[-10:].astype(float)
                log = lambda v: np.sign(v)*np.log1p(np.abs(v))
                features = np.c_[np.ones(10), log(stock/5), log(pipe/5), log(forecast/5),
                                 log(x[-40:-30].astype(float)), x[-30:-20].astype(float)/7,
                                 x[-20:-10].astype(float)/500, np.full(10, (29-day)/28),
                                 (stock.sum()-stock)/500, (pipe.sum(0)-pipe)/500,
                                 (forecast.sum()-forecast)/500, np.eye(10)]
                scores = features @ model.actor.weight.detach().numpy().T
                probabilities = np.zeros((10, 41))
                for sku in range(10):
                    physical = np.clip(center[sku]+np.arange(-2, 3), 0, 40)
                    actions, slots = np.unique(physical, return_index=True)
                    logits = scores[sku, slots]+np.log(np.where(actions == default[sku], .98, .02/(len(actions)-1)))
                    p = np.exp(logits-logits.max()); p /= p.sum()
                    probabilities[sku, actions] = p
                    if day >= 27:
                        probabilities[sku] = 0; probabilities[sku, default[sku]] = 1
                np.testing.assert_allclose(model.actor(torch.tensor(x)[None], [day]).probs.detach().numpy()[0],
                                           probabilities, rtol=2e-5, atol=2e-7)
                terms = row['info']['reward_terms']
                reward = terms['revenue']-sum(terms[k] for k in ['ordering_cost', 'holding_cost', 'stockout_cost', 'price_change_cost'])
                assert abs(reward-row['reward']) < 1e-8
                assert row['info']['total_on_hand'] <= 500+1e-8
                if day >= 27:
                    np.testing.assert_array_equal(action, default)
                for field in ['obs', 'state', 'action']:
                    row[field] = np.asarray(row[field])
        metrics = model.update(chunk)
        assert metrics['active_actor_rows'] == 208
        saved = torch.load(out/f'update_{update+1}.pt', weights_only=True)
        for key, value in model.actor.state_dict().items():
            torch.testing.assert_close(value, saved['actor'][key], rtol=0, atol=0)
        for key, value in model.critic.state_dict().items():
            torch.testing.assert_close(value, saved['critic'][key], rtol=0, atol=0)
        torch.testing.assert_close(model.generator.get_state(), saved['action_rng'], rtol=0, atol=0)
        for key in metrics:
            assert metrics[key] == summary['history'][update][key], key
    assert summary['projected_training_seconds'] == summary['seconds']/16*1536
    assert summary['budget_fits'] == (summary['projected_training_seconds'] <= 1800)
    write_json(out/'verification.json', dict(status='passed', verifier_sha256=digest(__file__),
               archived_days=448, reconstructed_updates=2, independent_probabilities=True,
               exact_policy_rng_and_checkpoint_replay=True, new_simulator_days=0,
               limitation='Archive policy/update replay; does not independently rerun simulator dynamics.'))
    print('Verified448 archived decisions, independent probabilities, reward arithmetic and two exact PPO updates.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', required=True)
    verify(parser.parse_args().output)
