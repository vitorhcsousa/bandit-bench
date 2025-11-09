from typing import Dict, List

import numpy as np
from numpy.typing import NDArray


class BanditMetrics:
    """Metrics calculator for evaluating contextual bandit performance.

    Tracks and computes various metrics including cumulative regret,
    average reward, and action distribution statistics.

    Attributes:
        rewards_history: List of rewards received over time.
        optimal_rewards_history: List of optimal rewards over time.
        actions_history: List of actions taken over time.
        regrets_history: List of regret values over time.
    """

    def __init__(self) -> None:
        """Initialize the metrics tracker."""
        self.rewards_history: List[float] = []
        self.optimal_rewards_history: List[float] = []
        self.actions_history: List[int] = []
        self.regrets_history: List[float] = []

    def record_step(
        self,
        action: int,
        reward: float,
        optimal_reward: float,
    ) -> None:
        """Record a single step of the bandit algorithm.

        Args:
            action: Action that was taken.
            reward: Reward received for the action.
            optimal_reward: Reward that would have been received for optimal action.
        """
        self.actions_history.append(action)
        self.rewards_history.append(reward)
        self.optimal_rewards_history.append(optimal_reward)

        regret = optimal_reward - reward
        self.regrets_history.append(regret)

    def get_cumulative_regret(self) -> NDArray[np.float64]:
        """Calculate cumulative regret over time.

        Cumulative regret is the sum of instantaneous regrets at each step.

        Returns:
            Array of cumulative regret values at each timestep.
        """
        return np.cumsum(self.regrets_history)

    def get_average_reward(self, window: int = 100) -> NDArray[np.float64]:
        """Calculate moving average of rewards.

        Args:
            window: Size of the moving average window.

        Returns:
            Array of moving average reward values.
        """
        if len(self.rewards_history) < window:
            return np.array([np.mean(self.rewards_history)])

        rewards_array = np.array(self.rewards_history)
        return np.convolve(
            rewards_array,
            np.ones(window) / window,
            mode="valid",
        )

    def get_total_reward(self) -> float:
        """Calculate total accumulated reward.

        Returns:
            Sum of all rewards received.
        """
        return float(np.sum(self.rewards_history))

    def get_average_regret(self) -> float:
        """Calculate average regret per step.

        Returns:
            Mean regret value.
        """
        return float(np.mean(self.regrets_history))

    def get_final_cumulative_regret(self) -> float:
        """Get the final cumulative regret value.

        Returns:
            Total cumulative regret.
        """
        return float(np.sum(self.regrets_history))

    def get_action_distribution(self) -> Dict[int, float]:
        """Calculate the distribution of actions taken.

        Returns:
            Dictionary mapping action indices to their selection frequencies.
        """
        total_actions = len(self.actions_history)
        if total_actions == 0:
            return {}

        unique_actions = set(self.actions_history)
        distribution = {
            action: self.actions_history.count(action) / total_actions for action in unique_actions
        }

        return distribution

    def get_optimal_action_rate(self, optimal_actions: List[int]) -> float:
        """Calculate the rate of selecting optimal actions.

        Args:
            optimal_actions: List of optimal actions for each timestep.

        Returns:
            Proportion of times the optimal action was selected.
        """
        if len(self.actions_history) != len(optimal_actions):
            raise ValueError("Length mismatch between actions and optimal actions")

        correct = sum(1 for a, opt_a in zip(self.actions_history, optimal_actions) if a == opt_a)
        return correct / len(self.actions_history)

    def get_summary(self) -> Dict[str, float]:
        """Get a summary of all key metrics.

        Returns:
            Dictionary containing all computed metrics.
        """
        return {
            "total_reward": self.get_total_reward(),
            "average_regret": self.get_average_regret(),
            "cumulative_regret": self.get_final_cumulative_regret(),
            "n_steps": len(self.rewards_history),
            "mean_reward": float(np.mean(self.rewards_history)),
            "std_reward": float(np.std(self.rewards_history)),
        }

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.rewards_history = []
        self.optimal_rewards_history = []
        self.actions_history = []
        self.regrets_history = []
