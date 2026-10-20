import copy
import numpy as np
import pytest
import torch
from src.algos.compact_residual import (CompactResidualActor, CompactResidualPPO,
                                       compact_features, episode_targets)
from src.algos.structured_imitation import InventoryPositionLinearPolicy
from src.algos.imitation_ordering import teacher_indices


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def observations():
    x = torch.zeros(6, 260)
    # Exact coefficients [5,-1,0] make these forecasts give centers0,1,20,39,40.
    x[:, -10:] = torch.tensor([0., 1., 20., 39., 40., 3.])[:, None]
    x[:, -20:-10] = 500
    return x


def test_prior_boundaries_unique_physical_actions_and_default():
    actor = CompactResidualActor([5, -1, 0]); x = observations()
    dist = actor(x, [1]*6)
    centers = InventoryPositionLinearPolicy(10, 7, [5, -1, 0]).predict(x)
    target = teacher_indices(x, 10, 7, cover=7.5).numpy()
    default = np.clip(target, np.maximum(centers-2, 0), np.minimum(centers+2, 40))
    np.testing.assert_array_equal(dist.probs.argmax(-1), default)
    for row in range(6):
        support = sorted(set(np.clip(centers[row, 0]+np.arange(-2, 3), 0, 40)))
        assert torch.where(dist.probs[row, 0] > 0)[0].tolist() == support
        for a in support:
            expected = .98 if a == default[row, 0] else .02/(len(support)-1)
            assert float(dist.probs[row, 0, a].detach()) == pytest.approx(expected, abs=2e-7)
    assert actor.weight.numel() == 165


def test_features_independent_indexing_and_competitor_timing():
    x = torch.arange(260, dtype=torch.float32)[None]
    f = compact_features(x, [8])[0, 2].numpy()
    log = lambda v: np.sign(v)*np.log1p(abs(v))
    p = x.numpy()[0, 10:80].reshape(10, 7)
    expected = np.r_[1, log(2/5), [log(v/5) for v in p[2]], log(252/5), log(222),
                     232/7, 242/500, 21/28, (45-2)/500,
                     (p.sum(0)-p[2])/500, (sum(range(250, 260))-252)/500, np.eye(10)[2]]
    np.testing.assert_allclose(f, expected, rtol=2e-7)
    y = x.clone(); y[0, 10] += 20; y[0, 11] -= 20
    other = compact_features(y, [8])[0, 2]
    assert not torch.equal(other[15:22], torch.tensor(f[15:22]))
    np.testing.assert_array_equal(other[2:9], f[2:9])


@pytest.mark.parametrize('day', [0, 29, 1.5, float('nan')])
def test_invalid_day(day):
    with pytest.raises(ValueError):
        compact_features(observations()[:1], [day])


def test_invalid_schema_and_nonfinite():
    with pytest.raises(ValueError):
        compact_features(torch.zeros(1, 259), [1])
    x = observations()[:1]; x[0, 1] = float('nan')
    with pytest.raises(ValueError):
        compact_features(x, [1])


def test_actual_sampling_log_probability_gradients_and_terminal():
    actor = CompactResidualActor([5, -1, 0]); x = observations()[2]
    action, lp = actor.act(x, 2, generator=torch.Generator().manual_seed(5))
    probabilities = actor(x[None], [2]).probs.detach().numpy()[0]
    expected = sum(np.log(probabilities[i, a]) for i, a in enumerate(action))
    got, entropy = actor.stats(x[None], [2], action[None])
    assert lp == pytest.approx(expected, abs=2e-6)
    assert float(got.detach()) == pytest.approx(lp)
    assert float((got-lp).exp().detach()) == pytest.approx(1)
    (-got-.01*entropy).sum().backward()
    assert torch.isfinite(actor.weight.grad).all() and actor.weight.grad.abs().sum() > 0
    before = probabilities.copy()
    with torch.no_grad():
        actor.weight[0, 0] += 2
    assert not np.array_equal(before, actor(x[None], [2]).probs.detach().numpy()[0])
    actor.zero_grad()
    a, _ = actor.act(x, 27)
    lp, entropy = actor.stats(x[None], [27], a[None])
    assert float(lp.detach()) == 0 and float(entropy.detach()) == 0
    (-lp-entropy).sum().backward()
    assert torch.count_nonzero(actor.weight.grad) == 0
    with pytest.raises(ValueError, match='support'):
        actor.stats(x[None], [2], np.zeros((1, 10), dtype=int))


def synthetic_episodes():
    actor = CompactResidualActor([5, -1, 0]); x = observations()[2].numpy()
    episodes = []
    for episode in range(8):
        rows = []
        for day in range(1, 29):
            action, lp = actor.act(x, day, True)
            rows.append(dict(obs=x.copy(), day=day, action=action, log_prob=lp,
                             state=np.array([.2, (29-day)/28], np.float32), value=.25,
                             reward=float(day+episode)))
        episodes.append(rows)
    return episodes


def test_gae_manual_terminal_and_episode_isolation():
    episodes = synthetic_episodes()
    adv, ret = episode_targets(episodes)
    expected = np.zeros(28)
    tail = 0
    for t in reversed(range(28)):
        delta = (t+1)/100 + (.99*.25 if t < 27 else 0) - .25
        tail = delta + (.99*.95*tail if t < 27 else 0)
        expected[t] = tail
    np.testing.assert_allclose(adv[:28], expected, rtol=2e-6, atol=1e-7)
    assert ret[27] == pytest.approx(.28)
    changed = copy.deepcopy(episodes); changed[1][0]['reward'] = 1e8
    np.testing.assert_array_equal(episode_targets(changed)[0][:28], adv[:28])
    with pytest.raises(ValueError):
        episode_targets([episodes[0][:-1]])


def test_update_checkpoint_optimizer_rng_roundtrip_and_schema(tmp_path):
    def model(seed=11):
        return CompactResidualPPO(1, [5, -1, 0], [str(i) for i in range(10)], {'test': 'synthetic'}, seed)
    m = model(); result = m.update(synthetic_episodes())
    assert result['daily_steps'] == 224 and result['active_actor_rows'] == 208
    assert 1 <= result['actor_epochs'] <= 4 and result['critic_epochs'] == 4
    assert torch.isfinite(m.actor.weight).all() and m.actor.weight.abs().sum() > 0
    path = tmp_path/'checkpoint.pt'; m.save(path)
    restored = model(); restored.load(path)
    for original, loaded in zip(m.actor.parameters(), restored.actor.parameters()):
        torch.testing.assert_close(original, loaded, rtol=0, atol=0)
    x = observations()[2]
    np.testing.assert_array_equal(m.actor.act(x, 1, generator=m.generator)[0],
                                  restored.actor.act(x, 1, generator=restored.generator)[0])
    # Optimizer moments matter: an identical next update must match exactly.
    m.update(synthetic_episodes()); restored.update(synthetic_episodes())
    torch.testing.assert_close(m.actor.weight, restored.actor.weight, rtol=0, atol=0)
    with pytest.raises(ValueError, match='specification'):
        model(22).load(path)
    data = torch.load(path, weights_only=True); data['variant'] = 'historical'
    torch.save(data, path)
    with pytest.raises(ValueError, match='variant'):
        model().load(path)
