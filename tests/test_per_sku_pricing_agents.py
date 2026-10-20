"""Per-SKU actor math, cadence, checkpoint schema and adapter verification."""
import copy
import inspect
import numpy as np
import pytest
import torch

from src.agents.pricing_agent import PricingPPOAgent
from src.algos.mappo import HeterogeneousMAPPO
from src.algos.independent_ppo import IndependentPPO
from src.algos.joint_ppo import JointPPO
from src.env.joint_env import JointRetailEnv
from src.env.pricing_env import PricingEnv
from src.env.forecast import RecentMeanForecaster


class Demand:
    def __init__(self):
        self.price = np.ones(10)
    def set_current_price(self, price):
        self.price = np.asarray(price).copy()
    def reset(self, rng):
        pass
    def __call__(self, day):
        return np.full(10, 2.0)


def env_for(cls, scope="per_sku"):
    kw = dict(pricing_scope=scope, episode_length_days=9,
              demand_source=Demand(), forecaster=RecentMeanForecaster(10),
              base_price=np.arange(1, 11), initial_inventory=20)
    if cls is PricingPPOAgent:
        return PricingEnv(**kw)
    return JointRetailEnv(**kw)


def make_agent(cls, scope="per_sku"):
    torch.set_num_threads(1)
    env = env_for(cls, scope)
    if cls is PricingPPOAgent:
        return cls(env, rollout_steps=4, minibatch_size=2, epochs_per_update=1)
    return cls(env, epochs=1)


def actor_heads(agent):
    if isinstance(agent, PricingPPOAgent):
        return agent.net.logits_heads
    if isinstance(agent, HeterogeneousMAPPO):
        return agent.pricing.heads
    if isinstance(agent, IndependentPPO):
        return agent.pricing[1].logits_heads
    return agent.net.logits_heads[:agent.n_price_heads]


def distinct_action(agent):
    with torch.no_grad():
        for i, head in enumerate(actor_heads(agent)):
            head.weight.zero_()
            head.bias.fill_(-50)
            head.bias[i % 5] = 50
    obs, _ = agent.env.reset(seed=3)
    if isinstance(agent, PricingPPOAgent):
        return agent.act(obs, greedy=True)[0]
    return agent.pricing_action(obs["pricing"], greedy=True)


CLASSES = [PricingPPOAgent, HeterogeneousMAPPO, IndependentPPO, JointPPO]


@pytest.mark.parametrize("cls", CLASSES)
def test_actor_has_independent_price_heads_and_stages_vector(cls):
    agent = make_agent(cls)
    action = distinct_action(agent)
    assert len(actor_heads(agent)) == 10
    np.testing.assert_array_equal(action, np.arange(10) % 5)
    if cls is PricingPPOAgent:
        _, _, _, _, info = agent.env.step(action)
        price = info["days"][0]["price"]
    else:
        agent.env.select_price(action)
        from src.env.interface import JointAction
        _, _, _, info = agent.env.step(JointAction(None, np.zeros(10, dtype=int)))
        price = info["price"]
    expected = agent.env.base_price * np.asarray(agent.env.config.pricing.multipliers)[action]
    np.testing.assert_allclose(price, expected)


@pytest.mark.parametrize("cls", [PricingPPOAgent, HeterogeneousMAPPO, IndependentPPO])
def test_price_vector_logprob_and_entropy_are_sums(cls):
    agent = make_agent(cls)
    obs, _ = agent.env.reset(seed=5)
    x = torch.as_tensor(obs if cls is PricingPPOAgent else obs["pricing"]).unsqueeze(0)
    a = torch.as_tensor((np.arange(10) % 5)[None, :], dtype=torch.long)
    if cls is HeterogeneousMAPPO:
        dists = agent.pricing(x)
        lp, entropy = agent.pricing.stats(x, a)
    else:
        ppo = agent.ppo if cls is PricingPPOAgent else agent.p_ppo
        dists, _ = ppo.net(x)
        lp, _, entropy = ppo.evaluate(x, a)
    torch.testing.assert_close(lp, sum(d.log_prob(a[:,i]) for i, d in enumerate(dists)))
    torch.testing.assert_close(entropy, sum(d.entropy() for d in dists))
    assert entropy.item() == pytest.approx(10 * np.log(5), abs=.01)


def test_joint_masks_all_ten_inactive_price_heads():
    agent = make_agent(JointPPO)
    x = torch.as_tensor(np.stack([agent.observation()] * 2))
    a = torch.zeros((2, 20), dtype=torch.long)
    lp, entropy, _ = agent.stats(x, a, torch.zeros(2))
    (lp.sum() + entropy.sum()).backward()
    for head in agent.net.logits_heads[:10]:
        assert torch.count_nonzero(head.weight.grad) == 0
        assert torch.count_nonzero(head.bias.grad) == 0
    assert torch.count_nonzero(agent.net.logits_heads[10].weight.grad) > 0
    with torch.no_grad():
        distributions, _ = agent.net(agent.transform(x))
        expected = sum(d.log_prob(a[:,i]) for i, d in enumerate(distributions))
        active, _, _ = agent.stats(x, a, torch.ones(2))
        torch.testing.assert_close(active, expected)


@pytest.mark.parametrize("cls", CLASSES)
def test_vector_checkpoint_roundtrip_and_schema_fail_before_mutation(cls, tmp_path):
    agent = make_agent(cls)
    expected = distinct_action(agent)
    path = tmp_path / "vector.pt"
    agent.save(path)
    data = torch.load(path, weights_only=True)
    assert data["action_schema"]["pricing_scope"] == "per_sku"
    assert data["action_schema"]["pricing_nvec"] == [5] * 10
    restored = make_agent(cls)
    restored.load(path)
    obs, _ = restored.env.reset(seed=3)
    actual = (restored.act(obs, greedy=True)[0] if cls is PricingPPOAgent
              else restored.pricing_action(obs["pricing"]))
    np.testing.assert_array_equal(actual, expected)
    shared = make_agent(cls, "shared")
    before = [p.detach().clone() for p in (
        shared.net.parameters() if cls in (PricingPPOAgent, JointPPO)
        else shared.pricing.parameters())]
    with pytest.raises(ValueError, match="incompatible pricing action schema"):
        shared.load(path)
    after = list(shared.net.parameters() if cls in (PricingPPOAgent, JointPPO)
                 else shared.pricing.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    data.pop("action_schema")
    torch.save(data, path)
    with pytest.raises(ValueError, match="legacy checkpoint"):
        restored.load(path)


@pytest.mark.parametrize("cls", CLASSES)
def test_legacy_metadata_missing_is_shared_only(cls, tmp_path):
    agent = make_agent(cls, "shared")
    path = tmp_path / "legacy.pt"
    agent.save(path)
    data = torch.load(path, weights_only=True)
    data.pop("action_schema")
    torch.save(data, path)
    make_agent(cls, "shared").load(path)
    with pytest.raises(ValueError, match="legacy checkpoint"):
        make_agent(cls).load(path)


@pytest.mark.parametrize("cls", [HeterogeneousMAPPO, IndependentPPO, JointPPO])
def test_small_synthetic_update_respects_weekly_vector_cadence(cls):
    agent = make_agent(cls)
    actions = []
    original = agent.env.select_price
    def record(action):
        actions.append(np.asarray(action).copy())
        return original(action)
    agent.env.select_price = record
    metrics = agent.train(1, seed=9)[0]
    assert metrics["pricing_decisions"] == 2
    assert len(actions) == 2 and all(a.shape == (10,) for a in actions)
    assert metrics["daily_steps"] == 9
    assert all(np.isfinite(v) for v in metrics.values())


def test_weekly_pricing_rollout_stores_all_heads():
    agent = make_agent(PricingPPOAgent)
    metrics = agent.train(1, seed=2)[0]
    assert np.isfinite(metrics["pi_loss"])
    evaluation = agent.evaluate(episodes=1, seed=4)
    assert len(evaluation["actions"][0]) == 2
    assert all(len(a) == 10 for a in evaluation["actions"][0])


def test_training_callbacks_and_research_metrics_preserve_vectors():
    from scripts.train_research import callbacks
    from src.eval.research import evaluate_episode
    agent = make_agent(HeterogeneousMAPPO)
    action = distinct_action(agent)
    order, price = callbacks("mappo", agent)
    obs, _ = agent.env.reset(seed=3)
    np.testing.assert_array_equal(price(obs["pricing"]), action)
    row = evaluate_episode(agent.env, order, price, seed=3)
    assert np.isfinite(row["profit"])
    assert row["discount_frequency"] == pytest.approx(.4)
    assert row["price_adjustment_frequency"] == 0


def test_evaluation_scope_reconstructs_legacy_and_requires_explicit_final():
    from scripts.evaluate_research import checkpoint_pricing_scope
    from scripts.train_clean_research import run_training
    assert inspect.signature(run_training).parameters["pricing_scope"].default == "per_sku"
    assert checkpoint_pricing_scope({}) == "shared"
    assert checkpoint_pricing_scope({"pricing_scope": "per_sku"}) == "per_sku"
    with pytest.raises(ValueError, match="disagrees"):
        checkpoint_pricing_scope({}, requested="per_sku")
    with pytest.raises(ValueError, match="explicit"):
        checkpoint_pricing_scope({}, requested="shared", final=True)


def test_joint_callbacks_return_only_order_heads_on_every_day():
    from src.experiments.phase4 import evaluate
    agent = make_agent(JointPPO)
    distinct_action(agent)
    def order(env, obs):
        action = agent.order_action(env, obs)
        assert action.shape == (10,)
        return action
    result = evaluate(agent.env, order, episodes=2, seed=7, price_policy=agent.pricing_action)
    assert np.isfinite(result["mean_daily_reward"])
