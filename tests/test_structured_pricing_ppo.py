import numpy as np
import torch
from src.algos.structured_pricing_ppo import StructuredPricingActor, pricing_features, critic_features
from src.algos.mappo import interval_gae
from src.algos.structured_pricing_ppo import StructuredPricingPPO
from scripts.verify_structured_pricing_ppo import independent_update


def test_initial_policy_and_local_inventory_response():
    x = np.zeros(187, np.float32)
    x[-20:-10] = 10
    x[150:160] = 2
    f = pricing_features(x, 1, np.full(10, 2))
    actor = StructuredPricingActor()
    np.testing.assert_array_equal(actor.act(f, greedy=True)[0], np.full(10, 4))
    np.testing.assert_allclose(actor.distribution(torch.tensor(f)).probs.detach(),
                               np.tile([.05, .05, .05, .05, .8], (10, 1)), atol=1e-7)
    x[:10] = 100
    assert np.all(pricing_features(x, 1, np.full(10, 2))[:, 1] > f[:, 1])
    assert critic_features(np.zeros(267), 28)[-1] == np.float32(1 / 28)


def test_weekly_returns_equal_discounted_daily_returns_at_lambda_one():
    rewards = np.arange(28, dtype=float)
    _, returns = interval_gae(rewards, [0, 0, 0, 0], [0, 7, 14, 21], gamma=.99, lam=1)
    expected = [sum(.99 ** k * r for k, r in enumerate(rewards[start:])) for start in [0, 7, 14, 21]]
    np.testing.assert_allclose(returns, expected, rtol=1e-6)


def test_independent_update_reconstruction():
    torch.set_num_threads(1)
    model, check = StructuredPricingPPO(3, 601), StructuredPricingPPO(3, 601)
    batch = []
    for e in range(3):
        decisions = []
        for week in range(4):
            features = np.ones((10, 7), np.float32) * (.1 + week)
            state = np.ones(4, np.float32) * (e + week)
            a, logp = model.actor.act(features)
            decisions.append(dict(features=features.tolist(), action=a.tolist(), logp=logp,
                                  state=state.tolist(), value=model.value(state)))
        batch.append(dict(decisions=decisions, rewards=list(range(e, e + 28))))
    model.update(batch)
    independent_update(check, batch)
    for name in ["actor", "critic"]:
        for key, value in getattr(model, name).state_dict().items():
            assert torch.equal(value, getattr(check, name).state_dict()[key])
