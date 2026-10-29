import numpy as np
import pytest
import torch

import config
from environment import PortfolioEnv
from networks import ActorCritic
from ppo import PPOAgent, RolloutBuffer


def _buffer(rewards, values, dones):
    b = RolloutBuffer()
    for r, v, d in zip(rewards, values, dones):
        b.add(None, None, None, None, 0.0, r, v, d)
    return b


def test_gae_matches_hand_computation():
    g, lam = config.GAMMA, config.GAE_LAMBDA
    b = _buffer([1.0, 2.0], [0.5, 1.0], [False, False])
    adv, ret = b.compute_gae(last_value=3.0)
    d1 = 2.0 + g * 3.0 - 1.0
    d0 = 1.0 + g * 1.0 - 0.5
    assert adv[1] == pytest.approx(d1)
    assert adv[0] == pytest.approx(d0 + g * lam * d1)
    assert ret == pytest.approx(adv + np.array([0.5, 1.0]))


def test_gae_does_not_bootstrap_through_terminal():
    b = _buffer([1.0], [0.5], [True])
    adv, _ = b.compute_gae(last_value=100.0)
    assert adv[0] == pytest.approx(1.0 - 0.5)


def test_time_limit_trajectory_bootstraps_from_critic(synthetic):
    features, prices, _ = synthetic
    agent = PPOAgent(config.N_ASSETS * config.N_FEATURES, config.N_ASSETS)
    env = PortfolioEnv(features, prices, is_train=True)
    _, steps, last_value = agent.collect_trajectory(env)
    assert steps == config.TRAJECTORY_LENGTH
    assert not agent.buffer.dones[-1]
    assert last_value != 0.0


def test_end_of_data_trajectory_is_terminal(synthetic):
    features, prices, _ = synthetic
    agent = PPOAgent(config.N_ASSETS * config.N_FEATURES, config.N_ASSETS)
    env = PortfolioEnv(features[:100], prices[:100], is_train=True)
    _, _, last_value = agent.collect_trajectory(env)
    assert agent.buffer.dones[-1] and last_value == 0.0


def test_actor_critic_outputs_valid_weights():
    model = ActorCritic(config.N_ASSETS * config.N_FEATURES, config.N_ASSETS)
    x = torch.randn(4, config.LOOKBACK_WINDOW, config.N_ASSETS * config.N_FEATURES)
    w0 = torch.full((4, config.N_ASSETS), 1 / config.N_ASSETS)
    weights, logits, log_prob, value = model.act(x, w0)
    assert weights.shape == (4, config.N_ASSETS)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-5)
    assert log_prob.shape == (4,) and value.shape == (4,)


def test_initial_policy_is_near_uniform_and_not_dead():
    torch.manual_seed(0)
    model = ActorCritic(config.N_ASSETS * config.N_FEATURES, config.N_ASSETS)
    x = torch.randn(8, config.LOOKBACK_WINDOW, config.N_ASSETS * config.N_FEATURES)
    w0 = torch.full((8, config.N_ASSETS), 1 / config.N_ASSETS)
    mean, _, _ = model(x, w0)
    assert mean.abs().max() < 0.5          # near-uniform start
    assert mean.std() > 0                  # but responsive to the input


def test_model_has_no_stochastic_layers():
    # Dropout would make rollout/update log-probs disagree and make backtests non-deterministic.
    model = ActorCritic(config.N_ASSETS * config.N_FEATURES, config.N_ASSETS)
    assert model.feature_extractor.lstm.dropout == 0.0
    x = torch.randn(2, config.LOOKBACK_WINDOW, config.N_ASSETS * config.N_FEATURES)
    w0 = torch.full((2, config.N_ASSETS), 1 / config.N_ASSETS)
    model.train()
    a, b = model(x, w0)[0], model(x, w0)[0]
    assert torch.equal(a, b)


def test_first_ppo_epoch_has_ratio_one(synthetic):
    features, prices, _ = synthetic
    agent = PPOAgent(config.N_ASSETS * config.N_FEATURES, config.N_ASSETS)
    env = PortfolioEnv(features, prices, is_train=True)
    agent.collect_trajectory(env)
    mw, cw, logits, old_lp, _, _ = agent.buffer.get_tensors(np.zeros(config.TRAJECTORY_LENGTH), np.zeros(config.TRAJECTORY_LENGTH), agent.device)
    with torch.no_grad():
        new_lp, _, _ = agent.model.evaluate(mw, cw, logits)
    assert torch.allclose(new_lp, old_lp, atol=1e-3)


def test_update_changes_parameters_and_stays_finite(synthetic):
    features, prices, _ = synthetic
    agent = PPOAgent(config.N_ASSETS * config.N_FEATURES, config.N_ASSETS)
    env = PortfolioEnv(features, prices, is_train=True)
    _, _, last_value = agent.collect_trajectory(env)
    before = [p.clone() for p in agent.model.parameters()]
    stats = agent.update(last_value)
    assert all(np.isfinite(v) for v in stats.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.model.parameters()))


def test_checkpoint_roundtrip_and_mismatch_detection(tmp_path):
    agent = PPOAgent(config.N_ASSETS * config.N_FEATURES, config.N_ASSETS)
    path = str(tmp_path / "m.pt")
    agent.save(path)

    other = PPOAgent(config.N_ASSETS * config.N_FEATURES, config.N_ASSETS)
    other.load(path)
    for a, b in zip(agent.model.parameters(), other.model.parameters()):
        assert torch.equal(a, b)

    wrong = PPOAgent(10 * config.N_FEATURES, 10)
    with pytest.raises(ValueError):
        wrong.load(path)
