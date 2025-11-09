"""Tests for bandit implementations."""

import numpy as np
from cb_comparison.bandits import PyTorchBandit, RiverBandit, VowpalWabbitBandit


def test_pytorch_bandit_initialization() -> None:
    """Test PyTorch bandit initialization."""
    bandit = PyTorchBandit(
        n_actions=5,
        input_dim=10,
        exploration_algorithm="epsilon-greedy",
    )

    assert bandit.n_actions == 5
    assert bandit.input_dim == 10
    assert bandit.exploration_algorithm == "epsilon-greedy"
    assert not bandit.is_fitted


def test_pytorch_bandit_predict() -> None:
    """Test PyTorch bandit prediction."""
    bandit = PyTorchBandit(n_actions=3, input_dim=5)
    context = np.random.randn(5)

    action, scores = bandit.predict(context)

    assert isinstance(action, int)
    assert 0 <= action < 3
    assert scores.shape == (3,)


def test_pytorch_bandit_update() -> None:
    """Test PyTorch bandit update."""
    bandit = PyTorchBandit(n_actions=3, input_dim=5)
    context = np.random.randn(5)

    bandit.update(context, action=1, reward=1.0)

    assert bandit.is_fitted
    assert bandit.action_counts[1] == 1


def test_pytorch_bandit_reset() -> None:
    """Test PyTorch bandit reset."""
    bandit = PyTorchBandit(n_actions=3, input_dim=5)
    context = np.random.randn(5)

    bandit.update(context, action=1, reward=1.0)
    assert bandit.is_fitted

    bandit.reset()
    assert not bandit.is_fitted
    assert np.all(bandit.action_counts == 0)


def test_river_bandit_initialization() -> None:
    """Test River bandit initialization."""
    bandit = RiverBandit(n_actions=4, exploration_algorithm="ucb")

    assert bandit.n_actions == 4
    assert bandit.exploration_algorithm == "ucb"
    assert len(bandit.models) == 4


def test_river_bandit_predict_update() -> None:
    """Test River bandit prediction and update."""
    bandit = RiverBandit(n_actions=3, exploration_algorithm="epsilon-greedy")
    context = np.random.randn(5)

    action, scores = bandit.predict(context)
    assert 0 <= action < 3

    bandit.update(context, action=action, reward=0.5)
    assert bandit.is_fitted


def test_vowpal_bandit_initialization() -> None:
    """Test Vowpal Wabbit bandit initialization."""
    bandit = VowpalWabbitBandit(n_actions=5, exploration_algorithm="epsilon-greedy")

    assert bandit.n_actions == 5
    assert bandit.exploration_algorithm == "epsilon-greedy"


def test_vowpal_bandit_predict_update() -> None:
    """Test Vowpal Wabbit prediction and update."""
    bandit = VowpalWabbitBandit(n_actions=3, exploration_algorithm="epsilon-greedy")
    context = np.random.randn(5)

    action, probs = bandit.predict(context)
    assert 0 <= action < 3

    bandit.update(context, action=action, reward=1.0)
    assert bandit.is_fitted


def test_exploration_algorithms() -> None:
    """Test different exploration algorithms."""
    algorithms = ["epsilon-greedy", "ucb", "thompson"]

    for algo in algorithms:
        bandit = PyTorchBandit(
            n_actions=3,
            input_dim=5,
            exploration_algorithm=algo,
        )
        context = np.random.randn(5)

        action, _ = bandit.predict(context)
        assert 0 <= action < 3
