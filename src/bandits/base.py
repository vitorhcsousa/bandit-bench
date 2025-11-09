from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class ContextualBandit(ABC):
    """Abstract base class for contextual bandit algorithms.

    This class defines the interface that all contextual bandit implementations
    must follow, ensuring consistent API across different libraries and approaches.

    Attributes:
        n_actions: Number of available actions.
        exploration_algorithm: Name of the exploration algorithm used.
        model_params: Dictionary of model-specific parameters.
    """

    def __init__(
        self,
        n_actions: int,
        exploration_algorithm: str,
        **model_params: Any,
    ) -> None:
        """Initialize the contextual bandit.

        Args:
            n_actions: Number of available actions/arms.
            exploration_algorithm: Name of exploration algorithm (e.g., 'epsilon-greedy', 'ucb').
            **model_params: Additional model-specific parameters.
        """
        self.n_actions = n_actions
        self.exploration_algorithm = exploration_algorithm
        self.model_params = model_params
        self._is_fitted = False

    @abstractmethod
    def predict(
        self,
        context: NDArray[np.float64],
    ) -> tuple[int, NDArray[np.float64]]:
        """Select an action based on the context.

        Args:
            context: Context feature vector of shape (n_features,) or (batch_size, n_features).

        Returns:
            tuple containing:
                - Selected action index
                - Action probabilities/scores for all actions
        """
        pass

    @abstractmethod
    def update(
        self,
        context: NDArray[np.float64],
        action: int,
        reward: float,
    ) -> None:
        """Update the model with observed reward.

        Args:
            context: Context feature vector that was used.
            action: Action that was taken.
            reward: Reward received for the action.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the bandit to initial state."""
        pass

    def get_info(self) -> dict[str, Any]:
        """Get information about the bandit configuration.

        Returns:
            Dictionary containing bandit configuration details.
        """
        return {
            "n_actions": self.n_actions,
            "exploration_algorithm": self.exploration_algorithm,
            "model_params": self.model_params,
            "is_fitted": self._is_fitted,
        }

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted with data.

        Returns:
            True if model has been updated at least once, False otherwise.
        """
        return self._is_fitted
