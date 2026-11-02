"""Frozen-fit producer reuse plus complete independent Fisher joint update audit."""
import copy
import json
from pathlib import Path
import math
import numpy as np
import torch

from outputs.context_curriculum_q_producer_fit_v1 import keys_contract, run_fit, set_training_calendar
from outputs.context_curriculum_q_independent_replay_v1 import replay_role_calibrated_q_control
from outputs.temperature_fit_engine import information_guard
from outputs.temperature_pilot_support import temperature
from outputs.conditional_credit_pilot_support import load
from outputs.verify_team_credit_pilot import optimizer_transition
from outputs.shared_price_fisher_independent_update_v1 import independent_update,independent_distribution,independent_kl
from outputs.temperature_pilot_replay import ledger_pair
from outputs.full_price_phase_profile_v1 import TIMINGS
from outputs.full_price_actor_v1 import dual_temperature_transition as temperature_transition
from src.experiments.execution_journal import digest


def coefficient(state, index):
    qq = state['q_weight.qq'][index]
    if qq <= 1e-12:
        return 0.
    return float((state['q_weight.qy'][index] / qq.clamp_min(1e-12)).clamp(0, 1).float())


def verify_transition(model, before, after, episodes, metrics, changes):
    assert torch.equal(after['rng'], episodes[-1]['actual_policy_rng'])
    assert before['variant'] == after['variant'] == model.variant
    assert before['design'] == after['design'] == model.design
    assert before['ordering_temperature'] == after['ordering_temperature']
    assert before['pricing_temperature'] == after['pricing_temperature'] == before['ordering_temperature']
    assert metrics['critic_epochs'] == 8 and metrics['role_scale_locks'] == 2
    assert torch.equal(after['model']['q_weight.observations'], before['model']['q_weight.observations'] + 1)
    assert metrics['calibration_observations'] == int(after['model']['q_weight.observations'][0])
    for index, role in enumerate(('pricing', 'ordering')):
        steps = metrics[role + '_actor_steps']
        assert type(steps) is int and 0 <= steps <= 4
        if role=='ordering':
            optimizer_transition(before['actors'][role],after['actors'][role],steps,.003)
        else:
            old_state,new_state=before['actors'][role],after['actors'][role]
            assert old_state['config']==new_state['config']
            attempts=sum(metrics['pricing_natural_'+str(i)+'_attempted'] for i in range(4))
            accepted=sum(metrics['pricing_natural_'+str(i)+'_accepted'] for i in range(4))
            trials=sum(metrics['pricing_natural_'+str(i)+'_trials'] for i in range(4))
            assert new_state['attempts']-old_state['attempts']==attempts
            assert new_state['accepted']-old_state['accepted']==accepted==steps
            assert new_state['rejected']-old_state['rejected']==attempts-accepted
            assert new_state['trials']-old_state['trials']==trials
            for i in range(4):
                stem='pricing_natural_'+str(i)+'_'
                if metrics[stem+'accepted']:
                    assert metrics[stem+'attempted'] and metrics[stem+'kl']<=.002
                    assert metrics[stem+'loss_after']<metrics[stem+'loss_before']
                if not metrics[stem+'attempted']:
                    assert all(metrics[stem+f]==0 for f in ('loss_before','loss_after','kl','gradient_norm','scale','trials','accepted','reason'))
        rows = [row for ep in episodes for row in ep['rows'][role]]
        roots = [root for ep in episodes for root in ep['roots'] if root['role'] == role]
        assert len(rows) == len(roots)
        features = torch.tensor(np.asarray([row['features'] for row in rows]), dtype=torch.float32)
        actions = torch.as_tensor(np.asarray([row['action'] for row in rows]), dtype=torch.long)
        old = torch.tensor([row['joint_logp'] for row in rows])
        with torch.no_grad():
            delta = getattr(model, role).distribution(features).log_prob(actions).sum(-1) - old
        assert np.isclose(float((delta.exp() - 1 - delta).mean()), metrics[role + '_kl'], rtol=5e-5, atol=3e-6)
        labels, qy, qq, squared = [], [], [], []
        for row, root in zip(rows, roots):
            coordinate = row['coordinate']
            assert row['q_beta_pre'] == coefficient(before['model'], index) == metrics[role + '_q_beta_pre']
            for entry in root['entries']:
                action = entry['action']
                p, y = float(np.float32(entry['probability'])), float(np.float32(entry['residual_return']))
                q = float(row['q_raw_pre'][coordinate, action])
                labels.append(y); qy.append(p * q * y); qq.append(p * q * q); squared.append(p * y * y)
        assert np.isclose(float(torch.tensor(labels, dtype=torch.float32).mean()), metrics[role + '_mc_target_mean'], rtol=1e-6, atol=1e-7)
        for field, values in (('qy', qy), ('qq', qq)):
            expect = float(before['model']['q_weight.' + field][index]) * .8 + .2 * math.fsum(values) / len(rows)
            assert np.isclose(float(after['model']['q_weight.' + field][index]), expect, rtol=2e-11, atol=2e-12)
        expected_scale = float(before['model']['role_scales.scales'][index])
        if not before['model']['role_scales.locked'][index]:
            expected_scale = max(.01, math.sqrt(math.fsum(squared) / len(rows)))
        assert np.isclose(float(after['model']['role_scales.scales'][index]), expected_scale, rtol=2e-11, atol=2e-12)
        assert bool(after['model']['role_scales.locked'][index])
        assert metrics[role + '_q_beta_post'] == coefficient(after['model'], index)
        assert metrics[role + '_locked_rms'] == float(after['model']['role_scales.scales'][index].float())
    optimizer_transition(before['critic_optimizer'], after['critic_optimizer'], 8, .001)
    for role in ('pricing', 'ordering', 'critics'):
        observed = any(not torch.equal(value, after['model'][key]) for key, value in before['model'].items() if key.startswith(role + '.'))
        assert observed == changes[role]
    for field in ('pricing.base_prices', 'pricing.phi', 'ordering.covers', 'pricing.temperature', 'ordering.temperature', 'pricing.log_prior', 'pricing.logit_gain'):
        assert torch.equal(before['model'][field], after['model'][field])
    assert all(torch.isfinite(value).all() for value in after['model'].values())


def independently_refit_update(model,before,after,episodes,metrics):
    # Same saved TRAIN batch, new independent numerical update, never a new fit bank.
    model.restore(copy.deepcopy(before));torch.set_rng_state(after['rng'])
    with TIMINGS.phase('independent_neural_updates'):
        computed=independent_update(model,episodes)
    actual=model.checkpoint()
    assert actual['model'].keys()==after['model'].keys()
    for key,value in actual['model'].items():
        if key=='pricing.weight':
            assert torch.allclose(value,after['model'][key],atol=1e-5,rtol=1e-5)
        else:assert torch.equal(value,after['model'][key]),key
    def exact(left,right):
        if isinstance(left,torch.Tensor):assert left.dtype==right.dtype and torch.equal(left,right)
        elif isinstance(left,dict):
            assert left.keys()==right.keys()
            for key in left:exact(left[key],right[key])
        elif isinstance(left,(list,tuple)):
            assert type(left)==type(right) and len(left)==len(right)
            for x,y in zip(left,right):exact(x,y)
        else:assert left==right
    for key in ('actors','critic_optimizer','rng'):exact(actual[key],after[key])
    assert computed.keys()==metrics.keys()
    for key,value in computed.items():
        if isinstance(value,(int,bool)):assert value==metrics[key],key
        else:assert np.isclose(value,metrics[key],atol=1e-5,rtol=1e-5),key
    model.restore(copy.deepcopy(after))



def audit_fit(folder, model, env, references, budget, stream, step, *, clone, stamp, calendar_starts):
    folder = Path(folder)
    receipt = json.loads((folder / 'fit_receipt.json').read_text(encoding='utf-8'))
    keys, positions, repeats = keys_contract(receipt['keys'], receipt['short_batch'])
    assert receipt['calendar_starts']==calendar_starts
    assert receipt['status'] == 'complete' and receipt['updates'] == len(keys) and receipt['repeats'] == repeats
    expected = {'final.pt'} | {f'{label}_{b:03d}.pt' for label in ('batch', 'after') for b in range(len(keys))}
    assert set(receipt['checkpoints']) == expected and set(receipt['data']) == {f'data_{b:03d}.pt' for b in range(len(keys))}
    files = dict(receipt['checkpoints'], **receipt['data'])
    assert all(digest(folder / path) == value for path, value in files.items())
    torch.manual_seed(receipt['torch_seed'])
    initial = load(folder / 'batch_000.pt')
    assert torch.equal(initial['rng'], torch.get_rng_state())
    assert all(torch.equal(value, initial['model'][key]) for key, value in model.state_dict().items())
    info = 'local' if model.mode == 'local' else 'mappo'
    main_days = 0
    with information_guard(env, info):
        for batch, (scenario_keys, position) in enumerate(zip(keys, positions)):
            before = load(folder / f'batch_{batch:03d}.pt')
            model.restore(before)
            assert before['ordering_temperature'] == before['pricing_temperature'] == temperature('annealed', position)
            saved = load(folder / f'data_{batch:03d}.pt')
            episodes, entry = saved['episodes'], saved['update']
            assert entry == receipt['history'][batch] and [ep['key'] for ep in episodes] == list(scenario_keys)
            branch_days, seen = 0, set()
            for ep in episodes:
                key = ep['key']
                set_training_calendar(env,key,calendar_starts)
                assert ep['whole_reference_rewards'] == references[key]
                context = dict(arm=info + '_economic_q', batch=batch, key=key, root_role=None, root_day=None)
                budget.context = dict(context)
                with TIMINGS.phase('independent_replay'):
                    checked = replay_role_calibrated_q_control(model, env, ep, budget.advance, seen, clone=clone, stamp=stamp)
                branch_days += checked['branch_days']
                for _ in range(28):
                    step = ledger_pair(stream, step, 'actual', context); main_days += 1
                for branch in checked['branch_plan']:
                    part = dict(context, root_role=branch['role'], root_day=branch['day'])
                    for _ in range(branch['days']):
                        step = ledger_pair(stream, step, 'branch', part); main_days += 1
            assert branch_days == entry['branch_days']
            after = load(folder / f'after_{batch:03d}.pt')
            assert torch.equal(after['rng'], torch.get_rng_state())
            independently_refit_update(model,before,after,episodes*repeats,entry['metrics'])
            verify_transition(model, before, after, episodes * repeats, entry['metrics'], entry['changes'])
            following = load(folder / (f'batch_{batch + 1:03d}.pt' if batch + 1 < len(keys) else 'final.pt'))
            tau = temperature('annealed', positions[batch + 1]) if batch + 1 < len(keys) else temperature('annealed', position)
            temperature_transition(after, following, tau)
        model.restore(load(folder / 'final.pt'))
    assert main_days == receipt['completed_days'] == receipt['reserved_days']
    assert all(digest(folder / path) == value for path, value in files.items())
    return dict(status='PASS', main_days=main_days, ledger_step=step, updates_checked=len(keys),
                checkpoints_checked=len(expected), sgd_refitted=True,independent_joint_updates_refitted=len(keys))
