"""Bounded matched teacher-only versus learner-state imitation diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from scripts.diagnose_teacher_ordering import measure, compare
from scripts.train_clean_research import TrainingWindows
from src.algos.imitation_ordering import ImitationAssistedMAPPO
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env


def acceptance(agreement, rows, teacher, initial):
    teacher_delta, initial_delta = compare(rows, teacher), compare(rows, initial)
    passed = (agreement >= .99 and np.mean([r['teacher_agreement'] for r in rows]) >= .99
              and teacher_delta['reward_ci95'][0] >= -1.
              and teacher_delta['fill_ci95'][0] >= -.005
              and initial_delta['reward_ci95'][0] > 0
              and initial_delta['fill_ci95'][0] >= -.005)
    return dict(passed=bool(passed), vs_teacher=teacher_delta, vs_initial=initial_delta)


def fit(actor, optimizer, x, y, epochs, generator, deadline):
    losses, steps = [], 0
    for _ in range(epochs):
        for batch in torch.randperm(len(x), generator=generator).split(128):
            if time.perf_counter() > deadline:
                raise TimeoutError('state-aggregation campaign deadline')
            distributions = actor(x[batch])
            loss = -torch.stack([d.log_prob(y[batch, i]) for i,d in enumerate(distributions)], -1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite imitation loss')
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), .5)
            optimizer.step(); steps += 1; losses.append(loss.item())
    return dict(mean_batch_loss=float(np.mean(losses)), optimizer_steps=steps)


@torch.no_grad()
def agreement(actor, x, y):
    predictions = torch.cat([torch.stack([d.probs.argmax(-1) for d in actor(batch)], -1)
                             for batch in x.split(128)])
    return float((predictions == y).float().mean())


def run(output, calibration):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); deadline = started + 900
    torch.set_num_threads(1)
    protocol = dict(variant='learner_state_aggregation_diagnostic_v1', seed=11,
        hypothesis='Training-only learner-visited teacher labels improve own-state imitation relative to matched teacher-only collection.',
        arms=['teacher_only', 'learner_states'], warmup_episodes=64,
        rounds=4, episodes_per_round=64, total_collection_episodes_per_arm=320,
        warmup_epochs=20, epochs_per_round=10, batch_size=128, lr=.0003,
        hidden=64, normalize_inputs=True, architecture_changed=False,
        training_holdout_episodes=16, holdout_seeds=list(range(410000,410016)),
        collection_seeds=list(range(400000,400320)),
        validation_seeds=list(range(40000,40020)), validation_days=[1886,1913],
        train_end_day=1885, days=28, scenario='S2', price_multiplier=1.2,
        pricing_scope='shared_fixed_fixture', capacity=500, billing='accepted_delivered_units',
        teacher='float32 actor-local cover6 prior center', calibration=str(Path(calibration).resolve()),
        calibration_sha256=hashlib.sha256(Path(calibration).read_bytes()).hexdigest(),
        max_seconds=900, held_out_test_used=False, checkpoint_selection='final round only; no validation during fitting',
        gates=dict(heldout_train_agreement=.99, validation_own_state_agreement=.99,
                   teacher_reward_ci_lower_min=-1., teacher_fill_ci_lower_min=-.005,
                   initial_reward_ci_lower_strict_min=0., initial_fill_ci_lower_min=-.005),
        conditional_residual=dict(max_updates=64, requires='learner-state equivalence gate AND projected greedy/stochastic teacher-equivalence',
            reward_ci_lower_strict_min=0., fill_ci_lower_min=-.005,
            comparators=['own_initial', 'teacher', 'constant', 'ss'], max_seconds=570),
        expansion='No automatic multiseed or learned-price expansion in this experiment')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    # Pin source before data collection; outputs cannot silently become another method.
    (out/'source_sha256.txt').write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        params = load_identifiable_training_calibration(calibration)
        train = TrainingWindows(make_research_env('S2', split='train', start=15, days=28,
                                    params=params, pricing_scope='shared'),28)
        validation = make_research_env('S2', split='validation', params=params, pricing_scope='shared')
        def measured(env, actor, seeds, label, collect=False):
            return measure(env, actor, seeds, label, collect=collect, deadline=deadline,
                           trace_path=out/'traces.jsonl')
        def dataset(actor, seeds, label):
            _,_,x,y = measured(train,actor,seeds,label,collect=True)
            return torch.tensor(np.asarray(x),dtype=torch.float32), torch.tensor(np.asarray(y),dtype=torch.long)
        hold_x, hold_y = dataset(None, protocol['holdout_seeds'], 'heldout_train_teacher')
        base_x, base_y = dataset(None, protocol['collection_seeds'][:64], 'warmup_teacher')
        torch.manual_seed(11)
        warm = ImitationAssistedMAPPO(train,normalize_inputs=True,target_kl=.02)
        warm.save(out/'random_initial.pt')
        initial_state = {k:v.detach().clone() for k,v in warm.replenishment.state_dict().items()}
        warm_metrics = fit(warm.replenishment,warm.optimizers[1],base_x,base_y,20,
                           torch.Generator().manual_seed(11),deadline)
        warm.save(out/'warmup.pt')
        warm_metrics['heldout_training_agreement'] = agreement(warm.replenishment,hold_x,hold_y)
        history, models = [], {}
        for arm in protocol['arms']:
            torch.manual_seed(11)
            model = ImitationAssistedMAPPO(train,normalize_inputs=True,target_kl=.02)
            model.load(out/'warmup.pt')
            x, y = base_x.clone(), base_y.clone()
            rng = torch.Generator().manual_seed(12011)
            for round_index in range(4):
                seeds = protocol['collection_seeds'][64*(round_index+1):64*(round_index+2)]
                new_x,new_y = dataset(model.replenishment if arm=='learner_states' else None,
                                      seeds,arm+'_collection_'+str(round_index+1))
                x, y = torch.cat([x,new_x]), torch.cat([y,new_y])
                metrics = fit(model.replenishment,model.optimizers[1],x,y,10,rng,deadline)
                metrics.update(arm=arm,round=round_index+1,labeled_states=len(x),
                    heldout_training_agreement=agreement(model.replenishment,hold_x,hold_y),
                    aggregate_training_agreement=agreement(model.replenishment,x,y))
                history.append(metrics)
                (out/'training_progress.json').write_text(json.dumps(history,indent=2))
                print(json.dumps(metrics),flush=True)
            model.save(out/(arm+'.pt'))
            models[arm] = model
        # Final-only evaluation; holdout observations never enter optimizer batches.
        seeds=protocol['validation_seeds']
        initial = ImitationAssistedMAPPO(validation,normalize_inputs=True,target_kl=.02)
        initial.replenishment.load_state_dict(initial_state)
        rows_by_label, traces_by_label = {}, {}
        for label, actor in [('initial',initial.replenishment),('warmup',warm.replenishment),
                             ('teacher',None),('constant','constant'),('ss','ss')]+[
                                 (arm,models[arm].replenishment) for arm in protocol['arms']]:
            rows,traces,_,_=measured(validation,actor,seeds,label)
            rows_by_label[label],traces_by_label[label]=rows,traces
        # Same fixed price and scenarios must produce identical realized demand.
        reference=[r['demand'] for r in traces_by_label['teacher']]
        for traces in traces_by_label.values():
            if [r['demand'] for r in traces] != reference:
                raise AssertionError('fixed-price demand pairing failed')
        gates={arm:acceptance(agreement(models[arm].replenishment,hold_x,hold_y),
                  rows_by_label[arm],rows_by_label['teacher'],rows_by_label['initial']) for arm in protocol['arms']}
        comparisons={arm:{ref:compare(rows_by_label[arm],rows_by_label[ref])
                    for ref in ['initial','warmup','teacher','constant','ss','teacher_only'] if ref!=arm}
                     for arm in protocol['arms']}
        for arm in protocol['arms']:
            restored=ImitationAssistedMAPPO(validation,normalize_inputs=True,target_kl=.02)
            restored.load(out/(arm+'.pt'))
            _,traces,_,_=measured(validation,restored.replenishment,seeds,arm+'_replay')
            if [dict(r,label=arm) for r in traces] != traces_by_label[arm]:
                raise AssertionError('checkpoint replay mismatch')
        summary=dict(protocol=protocol,warmup=warm_metrics,history=history,gates=gates,
            comparisons=comparisons,episode_rows=sum(rows_by_label.values(),[]),
            means={label:{key:float(np.mean([r[key] for r in rows]))
                         for key in ['reward_day','fill','teacher_agreement']} for label,rows in rows_by_label.items()},
            shared_warmup_episodes=64,additional_collection_episodes=512,
            total_labeled_collection_states=576*28,training_holdout_states=16*28,
            validation_episodes=140,replay_episodes=40,training_windows=train.windows,
            paired_demand_verified=True,checkpoint_replay_verified=True,
            seconds=time.perf_counter()-started,held_out_test_used=False,rl_executed=False,
            residual_eligible=gates['learner_states']['passed'])
        if hashlib.sha256(Path(calibration).read_bytes()).hexdigest()!=protocol['calibration_sha256']:
            raise AssertionError('calibration changed during run')
        (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        print(json.dumps(dict(gates=gates,means=summary['means'],seconds=summary['seconds']),indent=2),flush=True)
        return summary
    except Exception as exc:
        (out/'failed_status.json').write_text(json.dumps(dict(status='failed_partial',error=str(exc),held_out_test_used=False)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--calibration',default='outputs/calibration_correction/identifiable_v2/train_only_nb_identifiable_v2.json')
    args=p.parse_args();run(args.output,args.calibration)
