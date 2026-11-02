import numpy as np
import torch
from src.algos.parameter_policy_gradient import order_action
from src.algos.structured_ordering_ppo import StructuredOrderingActor, StructuredOrderingPPO, ordering_features, execute_cover
from scripts.verify_structured_ordering_ppo import independent_update, equal_state


def test_exact_deterministic_initial_actions_across_states():
    rng = np.random.default_rng(701)
    covers = rng.uniform(0, 16, 10)
    actor = StructuredOrderingActor(covers)
    for day in range(1, 29):
        obs = rng.uniform(0, 30, 260).astype(np.float32)
        latent, _ = actor.act(ordering_features(obs, day), greedy=True)
        np.testing.assert_array_equal(execute_cover(obs, latent), order_action(obs, day, covers))


def test_latent_likelihood_is_not_clipped_physical_likelihood():
    actor = StructuredOrderingActor(np.zeros(10))
    f = torch.ones(1, 10, 7)
    a, b = torch.full((1, 10), -1.), torch.full((1, 10), -2.)
    assert not torch.equal(actor.stats(f, a)[0], actor.stats(f, b)[0])
    obs = np.ones(260, np.float32)
    np.testing.assert_array_equal(execute_cover(obs, a[0].numpy()), execute_cover(obs, b[0].numpy()))


def test_independent_daily_ppo_update_and_optimizer_state():
    torch.set_num_threads(1)
    model = StructuredOrderingPPO(3, np.full(10, 7.5), 701)
    check = StructuredOrderingPPO(3, np.full(10, 7.5), 701)
    batch = []
    for e in range(3):
        rows = []
        for day in range(28):
            f = np.ones((10, 7), np.float32) * (.1 + day / 28)
            s = np.ones(4, np.float32) * (day + e)
            a, lp = model.actor.act(f)
            rows.append(dict(features=f.tolist(), state=s.tolist(), action=a.tolist(), logp=lp, value=model.value(s)))
        batch.append(dict(decisions=rows, rewards=list(range(e, e + 28))))
    model.update(batch); independent_update(check, batch)
    equal_state(model.state(), check.state())
