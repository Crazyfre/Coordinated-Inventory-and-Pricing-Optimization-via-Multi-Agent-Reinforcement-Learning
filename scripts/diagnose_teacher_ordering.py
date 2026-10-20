"""One bounded teacher-imitation then residual-RL gate; validation only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.algos.imitation_ordering import ImitationAssistedMAPPO, teacher_indices
from src.demand.identifiable_calibration import load_identifiable_training_calibration
from src.experiments.research import make_research_env
from scripts.train_clean_research import TrainingWindows
from src.env.interface import JointAction
from src.experiments.phase4 import heuristic


def measure(env, actor, seeds, label, collect=False, deadline=None, trace_path=None, greedy=True):
    summaries, traces, observations, targets = [], [], [], []
    for seed in seeds:
        torch.manual_seed(int(seed)+777)  # paired policy sampling, separate from env RNG
        env.reset(seed=seed)
        done, reward_sum, sold, demand_sum, agree, count = False, 0., 0., 0., 0, 0
        days = 0
        while not done:
            if deadline is not None and time.perf_counter() > deadline:
                raise TimeoutError('bounded teacher/evaluation deadline')
            if env.pricing_due:
                env.select_price(4)
            obs = env.replenishment_observation()
            target = teacher_indices(torch.as_tensor(obs), env.n, env.obs_pipe_horizon).numpy()
            if collect:
                observations.append(obs.copy()); targets.append(target.copy())
            if actor is None:
                action = target
            elif isinstance(actor, str):
                action = heuristic(env, kind=actor)
            else:
                action, _ = actor.act(obs, greedy=greedy)
            agree += int(np.sum(action == target)); count += env.n
            _, reward, done, info = env.step(JointAction(None, action))
            reward_sum += reward; sold += info['sales'].sum(); demand_sum += info['demand'].sum(); days += 1
            traces.append(dict(label=label, seed=int(seed), day=days, actions=np.asarray(action).tolist(),
                teacher=target.tolist(), demand=info['demand'].tolist(), sales=info['sales'].tolist(),
                price=info['price'].tolist(), reward=float(reward)))
            if trace_path is not None:
                with Path(trace_path).open('a') as handle:
                    handle.write(json.dumps(traces[-1])+'\n')
        summaries.append(dict(label=label, seed=int(seed), reward_day=float(reward_sum/days),
            fill=float(sold/demand_sum) if demand_sum else 1., teacher_agreement=agree/count))
    return summaries, traces, observations, targets


def compare(rows, baseline):
    if [r['seed'] for r in rows] != [r['seed'] for r in baseline]:
        raise ValueError('paired seeds required')
    rewards = np.array([r['reward_day']-b['reward_day'] for r,b in zip(rows,baseline)])
    fills = np.array([r['fill']-b['fill'] for r,b in zip(rows,baseline)])
    from scipy.stats import t
    width = float(t.ppf(.975,len(rewards)-1)*rewards.std(ddof=1)/np.sqrt(len(rewards)))
    fill_width = float(t.ppf(.975,len(fills)-1)*fills.std(ddof=1)/np.sqrt(len(fills)))
    return dict(reward_delta=float(rewards.mean()), reward_ci95=[float(rewards.mean()-width),float(rewards.mean()+width)],
                fill_delta=float(fills.mean()), fill_ci95=[float(fills.mean()-fill_width),float(fills.mean()+fill_width)],
                per_path_reward_delta=rewards.tolist())


def _run(output, calibration, seed=11, teacher_episodes=64, epochs=20, updates=64):
    if not (16 <= teacher_episodes <= 64 and 1 <= epochs <= 20 and 1 <= updates <= 64):
        raise ValueError('bounded teacher/epoch/update budgets required')
    torch.set_num_threads(1); torch.manual_seed(seed)
    started = time.perf_counter(); deadline = started + 570
    out = Path(output); out.mkdir(parents=True,exist_ok=False)
    params = load_identifiable_training_calibration(calibration)
    protocol = dict(variant='imitation_assisted_ordering_v1', price_multiplier=1.2, pricing_scope='shared_fixed_fixture',
        scenario='S2', days=28, seed=seed, initialization_seed=seed, teacher='float32 actor-local cover6 prior center', teacher_episodes=teacher_episodes, supervised_epochs=epochs,
        max_rl_updates=updates, calibration=str(Path(calibration).resolve()),
        calibration_sha256=hashlib.sha256(Path(calibration).read_bytes()).hexdigest(),
        calibration_status=params['status'], train_end=1885, validation_days=[1886,1913],
        validation_seeds=list(range(30000,30020)), held_out_test_used=False, torch_threads=1,
        billing='accepted_delivered_units', capacity=500, teacher_cover=6, lr=.0003,
        supervised_exact_agreement_min=.99, validation_exact_agreement_min=.99,
        imitation_reward_loss_max=1., imitation_fill_loss_max=.005,
        rl_reward_improvement_min=0., rl_fill_loss_max=.005, prior_budget_seconds=570)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    train = TrainingWindows(make_research_env('S2',split='train',start=15,days=28,params=params,pricing_scope='shared'),28)
    validation = make_research_env('S2',split='validation',start=1886,days=28,params=params,pricing_scope='shared')
    def measured(env, actor, seeds, label, **kwargs):
        return measure(env, actor, seeds, label, deadline=deadline, trace_path=out/'partial_traces.jsonl', **kwargs)
    dataset_seeds = list(range(110000,110000+teacher_episodes))
    holdout_episodes = max(4,teacher_episodes//4)
    _,_,obs,targets = measured(train,None,dataset_seeds,'teacher_dataset',collect=True)
    split = (teacher_episodes-holdout_episodes)*28
    x = torch.tensor(np.asarray(obs),dtype=torch.float32); y = torch.tensor(np.asarray(targets),dtype=torch.long)
    torch.manual_seed(seed)  # dataset pairing must not change declared initialization
    model = ImitationAssistedMAPPO(train,normalize_inputs=True,target_kl=.02)
    optimizer = model.optimizers[1]
    history = []
    for epoch in range(epochs):
        if time.perf_counter()>deadline: raise TimeoutError('bounded teacher experiment deadline')
        order = torch.randperm(split)
        losses = []
        for batch in order.split(128):
            dists = model.replenishment(x[batch])
            loss = -torch.stack([d.log_prob(y[batch,i]) for i,d in enumerate(dists)],-1).mean()
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.replenishment.parameters(),.5);optimizer.step()
            losses.append(loss.item())
        with torch.no_grad():
            predictions = torch.stack([d.probs.argmax(-1) for d in model.replenishment(x[split:])],-1)
            accuracy = float((predictions==y[split:]).float().mean())
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),heldout_training_action_agreement=accuracy))
    seeds = protocol['validation_seeds']
    teacher, teacher_trace, _, _ = measured(validation,None,seeds,'teacher')
    imitation, imitation_trace, _, _ = measured(validation,model.replenishment,seeds,'imitation')
    constant, constant_trace, _, _ = measured(validation,'constant',seeds,'constant')
    ss, ss_trace, _, _ = measured(validation,'ss',seeds,'ss')
    model.save(out/'imitation_checkpoint.pt')
    restored = ImitationAssistedMAPPO(validation,normalize_inputs=True,target_kl=.02)
    restored.load(out/'imitation_checkpoint.pt')
    replay, replay_trace, _, _ = measured(validation,restored.replenishment,seeds,'replay')
    if (imitation != [dict(r,label='imitation') for r in replay]
            or imitation_trace != [dict(r,label='imitation') for r in replay_trace]):
        raise AssertionError('imitation checkpoint replay mismatch')
    initial_difference = compare(imitation,teacher)
    imitation_gate = (accuracy>=.99 and np.mean([r['teacher_agreement'] for r in imitation])>=.99
        and initial_difference['reward_ci95'][0]>=-1 and initial_difference['fill_ci95'][0]>=-.005)
    rows = teacher+imitation+constant+ss
    traces = teacher_trace+imitation_trace+constant_trace+ss_trace
    summary = dict(protocol=protocol,supervised_history=history,training_windows=train.windows,
        labeled_train_states=len(obs),supervised_fit_states=split,supervised_holdout_states=len(obs)-split,
        imitation_gate=bool(imitation_gate),imitation_vs_teacher=initial_difference,
        checkpoint_roundtrip_verified=True,rl_executed=False,learning_gate=False,
        episode_rows=rows,reason='Teacher imitation gate failed; no RL refinement or multiseed expansion.')
    if imitation_gate:
        torch.manual_seed(seed)  # explicit reproducible critic initialization
        residual = ImitationAssistedMAPPO(train,residual=True,normalize_inputs=True,target_kl=.02)
        residual.replenishment.load_state_dict(model.replenishment.state_dict())
        initial, initial_trace,_,_ = measured(validation,residual.replenishment,seeds,'residual_initial')
        rows += initial; traces += initial_trace
        initial_teacher_difference = compare(initial,teacher)
        # Projection may change the controller; require its own fresh baseline gate.
        if (initial_teacher_difference['reward_ci95'][0]>=-1 and initial_teacher_difference['fill_ci95'][0]>=-.005):
            stochastic_initial, stochastic_initial_trace,_,_ = measured(validation,residual.replenishment,seeds,'stochastic_initial',greedy=False)
            training = []
            for update in range(updates):
                if time.perf_counter()>deadline: raise TimeoutError('bounded residual deadline')
                training += residual.train(1,seed=seed+update)
            trained, trained_trace,_,_ = measured(validation,residual.replenishment,seeds,'residual_trained')
            rows += trained;traces += trained_trace
            stochastic_trained, stochastic_trained_trace,_,_ = measured(validation,residual.replenishment,seeds,'stochastic_trained',greedy=False)
            rows += stochastic_initial+stochastic_trained
            traces += stochastic_initial_trace+stochastic_trained_trace
            own_difference, teacher_difference = compare(trained,initial),compare(trained,teacher)
            stochastic_difference = compare(stochastic_trained,stochastic_initial)
            stochastic_teacher_difference = compare(stochastic_trained,teacher)
            summary.update(rl_executed=True,rl_training=training,
                trained_vs_initial=own_difference,trained_vs_teacher=teacher_difference,
                stochastic_trained_vs_initial=stochastic_difference,
                stochastic_trained_vs_teacher=stochastic_teacher_difference,
                learning_gate=bool(own_difference['reward_ci95'][0]>0 and teacher_difference['reward_ci95'][0]>0
                    and own_difference['fill_ci95'][0]>=-.005 and teacher_difference['fill_ci95'][0]>=-.005
                    and stochastic_difference['reward_ci95'][0]>0 and stochastic_difference['fill_ci95'][0]>=-.005
                    and stochastic_teacher_difference['reward_ci95'][0]>0 and stochastic_teacher_difference['fill_ci95'][0]>=-.005),
                reason='Single bounded residual study; superiority requires positive paired reward lower bounds against initial and teacher.')
            residual.save(out/'residual_checkpoint.pt')
            residual_restored=ImitationAssistedMAPPO(validation,residual=True,normalize_inputs=True,target_kl=.02)
            residual_restored.load(out/'residual_checkpoint.pt')
            restored_rows, restored_trace,_,_=measured(validation,residual_restored.replenishment,seeds,'residual_replay')
            if (trained != [dict(r,label='residual_trained') for r in restored_rows]
                    or trained_trace != [dict(r,label='residual_trained') for r in restored_trace]):
                raise AssertionError('residual checkpoint trace mismatch')
            summary['residual_checkpoint_roundtrip_verified']=True
            summary['residual_checkpoint_replay_episodes']=len(restored_rows)
        else:
            summary['reason']='Residual projection baseline fails teacher equivalence; no RL.'
    summary.update(episode_rows=rows,evaluation_episodes=len(rows),checkpoint_replay_episodes=len(replay),
                   seconds=time.perf_counter()-started,held_out_test_used=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    with (out/'traces.jsonl').open('w') as handle:
        for row in traces: handle.write(json.dumps(row)+'\n')
    print(json.dumps({k:summary[k] for k in ('imitation_gate','rl_executed','learning_gate','seconds','reason')},indent=2))
    return summary


def run(output, calibration, seed=11, teacher_episodes=64, epochs=20, updates=64):
    if not (16 <= teacher_episodes <= 64 and 1 <= epochs <= 20 and 1 <= updates <= 64):
        raise ValueError('bounded teacher/epoch/update budgets required')
    if Path(output).exists():
        raise ValueError('fresh output directory required')
    try:
        return _run(output,calibration,seed,teacher_episodes,epochs,updates)
    except Exception as error:
        if Path(output).exists():
            (Path(output)/'failed_status.json').write_text(json.dumps(dict(status='failed_partial',error=str(error),held_out_test_used=False),indent=2))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--calibration',default='outputs/calibration_correction/identifiable_v2/train_only_nb_identifiable_v2.json')
    p.add_argument('--seed',type=int,default=11)
    p.add_argument('--teacher-episodes',type=int,default=64)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--updates',type=int,default=64)
    a=p.parse_args();run(a.output,a.calibration,a.seed,a.teacher_episodes,a.epochs,a.updates)
