"""Uninstalled full32x16fit/artifact/audit foundation; no automatic execution."""
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from outputs.full_price_coordinate_q_mappo_v1 import FullPriceCoordinateQMAPPO
from outputs.full_price_q_independent_replay_v1 import replay_full_price_q_control
from outputs.temperature_fit_engine import information_guard
from outputs.temperature_pilot_support import temperature
from outputs.conditional_credit_pilot_support import load, calendar
from outputs.fresh_support_pilot_support import training_collect
from outputs.verify_team_credit_pilot import optimizer_transition
from outputs.temperature_pilot_replay import ledger_pair
from outputs.full_price_phase_profile_v1 import TIMINGS
from outputs.full_price_actor_v1 import dual_temperature_transition as temperature_transition
from src.experiments.execution_journal import atomic_json, digest


def keys_contract(keys, short_batch):
    keys=tuple(tuple(row) for row in keys)
    if any(type(k) is not int or k<0 for row in keys for k in row) or len({k for row in keys for k in row})!=sum(map(len,keys)):
        raise ValueError('Distinct nonnegative integer scenario keys required')
    if short_batch:
        if len(keys)!=4 or any(len(row)!=2 for row in keys):raise ValueError('Short proof requires four paired episodes')
        return keys,(0,8,16,31),8
    if len(keys)!=32 or any(len(row)!=16 for row in keys):raise ValueError('Full32x16 required')
    return keys,tuple(range(32)),1


def set_training_calendar(env,key,starts):
    if str(key) not in starts or type(starts[str(key)]) is not int or not 15<=starts[str(key)]<=1858:
        raise ValueError('Complete prospectively registered legal TRAIN calendar required')
    if hasattr(env,'sim'):env.sim.start_day_index=starts[str(key)]


def run_fit(folder, model, env, references, keys, torch_seed, budget, *, short_batch=False, calendar_starts):
    keys, positions, repeats = keys_contract(keys, short_batch)
    if not all(k in references for row in keys for k in row):
        raise ValueError('same-key strong references required')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    before_days = budget.completed
    start = time.perf_counter()
    information = 'local' if model.mode == 'local' else 'mappo'
    histories, checkpoints, data_files = [], {}, {}
    torch.manual_seed(torch_seed)
    with information_guard(env, information):
        for batch, (scenario_keys, position) in enumerate(zip(keys, positions)):
            model.ordering.set_temperature(temperature('annealed', position))
            model.pricing.set_temperature(temperature('annealed', position))
            name = f'batch_{batch:03d}.pt'
            torch.save(model.checkpoint(), folder / name)
            checkpoints[name] = digest(folder / name)
            episodes = []
            for key in scenario_keys:
                set_training_calendar(env,key,calendar_starts)
                budget.context = dict(arm=information + '_economic_q', batch=batch, key=key, root_role=None, root_day=None)
                with TIMINGS.phase('collection'):
                    ep = training_collect(model, 'integrated', env, key, references[key], budget)
                ep['key'] = key
                episodes.append(ep)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            with TIMINGS.phase('neural_updates'):
                metrics = model.update(episodes * repeats)
            name = f'after_{batch:03d}.pt'
            torch.save(model.checkpoint(), folder / name)
            checkpoints[name] = digest(folder / name)
            changes = {role: any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items()
                                if k.startswith(role + '.')) for role in ('pricing', 'ordering', 'critics')}
            entry = dict(batch=batch, position=position, keys=list(scenario_keys), metrics=metrics,
                         changes=changes, branch_days=sum(ep['reference_days'] for ep in episodes),
                         temperature=float(model.ordering.temperature))
            name = f'data_{batch:03d}.pt'
            torch.save(dict(episodes=episodes, update=entry), folder / name)
            data_files[name] = digest(folder / name)
            histories.append(entry)
            atomic_json(folder / 'progress.json', dict(batch=batch, completed_days=budget.completed,
                reserved_days=budget.reserved, seconds=time.perf_counter() - start))
        torch.save(model.checkpoint(), folder / 'final.pt')
        checkpoints['final.pt'] = digest(folder / 'final.pt')
    assert budget.pending is None
    receipt = dict(status='complete', information=information, short_batch=short_batch, calendar_starts=calendar_starts,
        keys=[list(row) for row in keys], torch_seed=torch_seed, repeats=repeats,
        updates=len(keys), physical_training_episodes=sum(map(len, keys)),
        completed_days=budget.completed - before_days, reserved_days=budget.completed - before_days,
        checkpoints=checkpoints, data=data_files, history=histories, seconds=time.perf_counter() - start)
    atomic_json(folder / 'fit_receipt.json', receipt)
    return receipt
