"""Tests for the dataset module."""

import numpy as np

from cb_comparison.data import MessageFeedbackDataset


def test_dataset_initialization() -> None:
    """Test dataset initialization with default parameters."""
    dataset = MessageFeedbackDataset(n_actions=5, context_dim=10)

    assert dataset.n_actions == 5
    assert dataset.context_dim == 10
    assert len(dataset.actions) == 5
    assert dataset.action_weights.shape == (5, 10)


def test_generate_context() -> None:
    """Test context generation."""
    dataset = MessageFeedbackDataset(n_actions=3, context_dim=5, random_seed=42)

    contexts = dataset.generate_context(n_samples=10)

    assert contexts.shape == (10, 5)
    assert np.allclose(np.linalg.norm(contexts, axis=1), 1.0)


def test_get_optimal_action() -> None:
    """Test optimal action selection."""
    dataset = MessageFeedbackDataset(n_actions=3, context_dim=5, random_seed=42)
    context = dataset.generate_context(1)[0]

    optimal_action = dataset.get_optimal_action(context)

    assert isinstance(optimal_action, int)
    assert 0 <= optimal_action < 3


def test_get_reward_deterministic() -> None:
    """Test deterministic reward generation."""
    dataset = MessageFeedbackDataset(n_actions=3, context_dim=5, random_seed=42)
    context = dataset.generate_context(1)[0]

    reward1 = dataset.get_reward(context, action=0, deterministic=True)
    reward2 = dataset.get_reward(context, action=0, deterministic=True)

    assert reward1 == reward2


def test_get_reward_stochastic() -> None:
    """Test stochastic reward generation."""
    dataset = MessageFeedbackDataset(n_actions=3, context_dim=5, random_seed=42)
    context = dataset.generate_context(1)[0]

    rewards = [dataset.get_reward(context, action=0) for _ in range(100)]

    assert len(set(rewards)) > 1


def test_generate_batch() -> None:
    """Test batch generation."""
    dataset = MessageFeedbackDataset(n_actions=4, context_dim=8, random_seed=42)

    contexts, actions, rewards = dataset.generate_batch(batch_size=20)

    assert contexts.shape == (20, 8)
    assert actions.shape == (20,)
    assert rewards.shape == (20,)
    assert all(0 <= a < 4 for a in actions)
