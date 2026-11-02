"""Independent fixed-actor trajectories, normalized critic updates and metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical, Normal
from src.algos.structured_pricing_ppo import StructuredPricingPPO
from src.algos.structured_ordering_ppo import StructuredOrderingPPO
from scripts.verify_parameter_confirmation import episodes, check_episode
from scripts.verify_structured_ordering_ppo import equal_state
from scripts.train_compact_residual import digest, write_json


def reconstruct_fit(critic, data, history):
    x = np.asarray([s for e in data for s in e['states']], np.float32)
    y = np.asarray([t for e in data for t in e['targets']], float)
    mean, std = float(np.mean(y)), max(float(np.std(y)), 1e-6)
    target = torch.tensor((y - mean) / std, dtype=torch.float32)
    optimizer = torch.optim.Adam(critic.parameters(), lr=.003)
    for epoch, expected in enumerate(history):
        assert expected['epoch'] == epoch + 1
        loss = .5 * torch.mean((critic(torch.tensor(x)).squeeze(-1) - target)**2)
        assert float(loss.detach()) == expected['loss']
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.)
        optimizer.step()
        equal_state(critic.state_dict(), expected['critic'])
        equal_state(optimizer.state_dict(), expected['optimizer'])
    t = x[:, -1].astype(float)
    coefficients = np.linalg.lstsq(np.column_stack((np.ones(len(t)), t, t*t)), y, rcond=None)[0]
    return dict(mean=mean, std=std, time_coefficients=coefficients.tolist())


def verify(output):
    out = Path(output); load = lambda name: json.loads((out / name).read_text())
    manifest, summary = load('manifest.json'), load('summary.json')
    assert summary['status'] == 'complete'; torch.set_num_threads(1)
    for p, h in manifest['hashes'].items(): assert digest(p) == h, p
    for p, h in summary['archive_hashes'].items(): assert digest(out / p) == h, p
    count = 0
    for kind, first, second in [('pricing', 2270000, 2290000), ('ordering', 2280000, 2300000)]:
        root = Path(f'outputs/structured_mappo/{kind}_ppo_pilot_v1')
        prior = json.loads((root / 'manifest.json').read_text()); covers = np.array(prior['covers'])
        base = np.array(prior['base_prices'])
        model = (StructuredPricingPPO(prior['state_dim'], 601) if kind == 'pricing' else
                 StructuredOrderingPPO(prior['state_dim'], covers, 701))
        original = torch.load(root / 'initial.pt', weights_only=True, map_location='cpu')
        model.actor.load_state_dict(original['actor']); model.critic.load_state_dict(original['critic'])
        data_sets = {}; replay = None
        for split, start_key, number in [('fit', first, 128), ('assessment', second, 64)]:
            rebuilt = []; keys = []
            for (label, key), trace in episodes(out / f'{kind}_{split}.jsonl'):
                assert label == kind; keys.append(key); torch.manual_seed(key + 400000)
                start = int(np.random.default_rng(np.random.SeedSequence([key, 314159])).integers(15, 1859))
                check_episode(trace, start=start)
                states, targets = [], []
                rewards = np.array([r['reward'] for r in trace]) / 100
                prices = base * 1.2
                for i, row in enumerate(trace):
                    d = row['decision']; day = i + 1; obs = np.array(row['obs'], np.float32)
                    if kind == 'pricing' and i % 7:
                        assert d is None
                    else:
                        raw = np.array(d['raw_state'], np.float32)
                        state = np.r_[np.sign(raw) * np.log1p(np.abs(raw)) / 5, (29 - day) / 28].astype(np.float32)
                        np.testing.assert_array_equal(state, d['state']); assert d['value'] == 0.
                        states.append(state.tolist())
                        targets.append(float(sum(.99 ** k * r for k, r in enumerate(rewards[i:]))))
                        if kind == 'pricing':
                            x = np.array(d['obs'], np.float32); forecast = x[-20:-10]
                            f = np.stack([np.ones(10), np.clip(x[:10]/(1+7*forecast),0,3),
                                np.log1p(np.maximum(forecast,0))/5,
                                np.clip(x[10:150].reshape(10,14)[:,-7:].mean(1)/(1+forecast),0,3),
                                x[150:160]/base.astype(np.float32)-1, np.full(10,(29-day)/28),
                                np.clip(x[-10:]/500,0,1)],-1).astype(np.float32)
                            dist = Categorical(logits=torch.einsum('if,ifk->ik',torch.tensor(f),model.actor.weight))
                        else:
                            np.testing.assert_array_equal(raw[:260],obs)
                            pipe=obs[10:80].reshape(10,7); forecast=obs[-10:]
                            f=np.stack([np.ones(10),np.clip(obs[:10]/(1+forecast)/20,0,2),
                                np.clip(pipe.sum(1)/(1+forecast)/20,0,2),pipe[:,3:].sum(1)/(1+pipe.sum(1)),
                                np.log1p(np.maximum(forecast,0))/5,np.full(10,(29-day)/28),
                                np.clip(obs[-20:-10]/500,0,1)],-1).astype(np.float32)
                            dist=Normal(model.actor.covers+(torch.tensor(f)*model.actor.weight).sum(-1),.75)
                        np.testing.assert_array_equal(f,d['features'])
                        with torch.no_grad():
                            action=dist.sample(); assert float(dist.log_prob(action).sum()) == d['logp']
                        np.testing.assert_array_equal(action.numpy(),d['action'])
                        if kind == 'pricing': prices=base*np.array([.8,.9,1.,1.1,1.2])[action.numpy()]
                    c=np.array(covers if kind=='pricing' else d['action'],np.float32)
                    physical=np.ceil(np.maximum(c.clip(0,20)*obs[-10:]-(obs[:10]+obs[10:80].reshape(10,7).sum(1)),0)/5).clip(0,40).astype(int)
                    np.testing.assert_array_equal(physical,row['actions'])
                    np.testing.assert_allclose(prices,row['info']['price'],atol=1e-12,rtol=0)
                rebuilt.append(dict(states=states,targets=targets)); count+=1
                if split=='assessment' and key==second: replay=trace
            assert keys==list(range(start_key,start_key+number))
            assert rebuilt==load(f'{kind}_{split}_data.json'); data_sets[split]=rebuilt
        repeats=list(episodes(out/f'{kind}_replay.jsonl'))
        assert len(repeats)==1 and repeats[0][0]==(kind,second) and repeats[0][1]==replay; count+=1
        history=torch.load(out/f'{kind}_history.pt',weights_only=True,map_location='cpu'); assert len(history)==200
        norm=reconstruct_fit(model.critic,data_sets['fit'],history)
        assert norm==load(f'{kind}_normalization.json')
        equal_state(model.critic.state_dict(),torch.load(out/f'{kind}_critic.pt',weights_only=True,map_location='cpu'))
        for p,h in load(f'{kind}_assessment_freeze.json').items(): assert digest(out/p)==h
        data=data_sets['assessment']; x=np.asarray([s for e in data for s in e['states']],np.float32)
        y=np.asarray([t for e in data for t in e['targets']]); t=x[:,-1].astype(float)
        with torch.no_grad(): learned=model.critic(torch.tensor(x)).squeeze(-1).numpy()*norm['std']+norm['mean']
        predicted=dict(learned=learned,original=np.zeros(len(y)),constant=np.full(len(y),norm['mean']),
            time=np.column_stack((np.ones(len(t)),t,t*t))@np.array(norm['time_coefficients']))
        saved_predictions=load(f'{kind}_predictions.json'); result=summary['results'][kind]
        mse={}; by_episode={}
        for label,pred in predicted.items():
            np.testing.assert_array_equal(pred,saved_predictions[label]); err=pred-y
            mse[label]=float(np.mean(err*err)); by_episode[label]=(err*err).reshape(64,-1).mean(1)
            expected=dict(mse=mse[label],rmse=float(np.sqrt(mse[label])),bias=float(err.mean()),
                explained_variance=float(1-np.var(err)/np.var(y)))
            for metric,value in expected.items(): assert abs(value-result['controllers'][label][metric])<1e-10
        draws=np.random.default_rng(20261027).integers(64,size=(10000,64))
        for ref in ['constant','time','original']:
            d=by_episode['learned']-by_episode[ref]; expected=result['mse_differences'][ref]
            assert abs(d.mean()-expected['mean'])<1e-10
            np.testing.assert_allclose(np.quantile([d[i].mean() for i in draws],[.025,.975]),expected['ci95'],atol=1e-10,rtol=0)
        passed=(1-np.var(learned-y)/np.var(y)>=.2) and all(mse['learned']<=.9*mse[ref] for ref in ['constant','time'])
        assert bool(passed)==result['passed']
    assert count==summary['episodes']==386 and summary['days']==10808
    assert summary['seconds']<=900 and not summary['test_scoring']
    write_json(out/'verification.json',dict(status='passed',episodes=386,independent_critic_updates=400,
        frozen_actor_actions_targets_optimizers_metrics_replays=True,new_simulator_days=0,verifier_sha256=digest(__file__)))
    print('Critic calibration screen verified.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);verify(p.parse_args().output)
