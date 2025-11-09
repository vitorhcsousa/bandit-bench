"""Tests for evaluation metrics."""

import pytest
from cb_comparison.evaluation import BanditMetrics


def test_metrics_initialization() -> None:
    """Test metrics initialization."""
    metrics = BanditMetrics()

    assert len(metrics.rewards_history) == 0
    assert len(metrics.actions_history) == 0
    assert len(metrics.regrets_history) == 0


def test_record_step() -> None:
    """Test recording a single step."""
    metrics = BanditMetrics()

    metrics.record_step(action=1, reward=0.8, optimal_reward=1.0)

    assert len(metrics.actions_history) == 1
    assert len(metrics.rewards_history) == 1
    assert len(metrics.regrets_history) == 1
    assert metrics.regrets_history[0] == 0.2


def test_cumulative_regret() -> None:
    """Test cumulative regret calculation."""
    metrics = BanditMetrics()

    metrics.record_step(action=0, reward=0.5, optimal_reward=1.0)
    metrics.record_step(action=1, reward=0.8, optimal_reward=1.0)
    metrics.record_step(action=2, reward=1.0, optimal_reward=1.0)

    cumulative_regret = metrics.get_cumulative_regret()

    assert len(cumulative_regret) == 3
    assert cumulative_regret[0] == 0.5
    assert cumulative_regret[1] == 0.7
    assert cumulative_regret[2] == 0.7


def test_total_reward() -> None:
    """Test total reward calculation."""
    metrics = BanditMetrics()

    metrics.record_step(action=0, reward=0.5, optimal_reward=1.0)
    metrics.record_step(action=1, reward=0.8, optimal_reward=1.0)

    assert metrics.get_total_reward() == 1.3


def test_average_regret() -> None:
    """Test average regret calculation."""
    metrics = BanditMetrics()

    metrics.record_step(action=0, reward=0.6, optimal_reward=1.0)
    metrics.record_step(action=1, reward=0.8, optimal_reward=1.0)

    avg_regret = metrics.get_average_regret()

    assert avg_regret == pytest.approx(0.3)


def test_action_distribution() -> None:
    """Test action distribution calculation."""
    metrics = BanditMetrics()

    metrics.record_step(action=0, reward=0.5, optimal_reward=1.0)
    metrics.record_step(action=0, reward=0.6, optimal_reward=1.0)
    metrics.record_step(action=1, reward=0.8, optimal_reward=1.0)
    metrics.record_step(action=2, reward=0.9, optimal_reward=1.0)

    distribution = metrics.get_action_distribution()

    assert distribution[0] == 0.5
    assert distribution[1] == 0.25
    assert distribution[2] == 0.25


def test_get_summary() -> None:
    """Test summary statistics."""
    metrics = BanditMetrics()

    for i in range(10):
        metrics.record_step(action=i % 3, reward=0.5 + i * 0.05, optimal_reward=1.0)

    summary = metrics.get_summary()

    assert "total_reward" in summary
    assert "average_regret" in summary
    assert "cumulative_regret" in summary
    assert summary["n_steps"] == 10


def test_reset() -> None:
    """Test metrics reset."""
    metrics = BanditMetrics()

    metrics.record_step(action=0, reward=0.5, optimal_reward=1.0)
    assert len(metrics.rewards_history) == 1

    metrics.reset()
    assert len(metrics.rewards_history) == 0
    assert len(metrics.actions_history) == 0
