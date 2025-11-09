from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray
from river import linear_model, optim

from .base import ContextualBandit


class RiverBandit(ContextualBandit):
    """River-based contextual bandit using online learning.

    Implements contextual bandits using River's online learning models with
    support for epsilon-greedy, UCB, and Thompson sampling exploration.

    Attributes:
        models: List of River models, one per action.
        epsilon: Exploration parameter for epsilon-greedy.
        ucb_alpha: Confidence parameter for UCB.
        action_counts: Count of times each action was selected.
        action_values: Running average of rewards per action.
    """

    def __init__(
        self,
        n_actions: int,
        exploration_algorithm: str = "epsilon-greedy",
        learning_rate: float = 0.01,
        epsilon: float = 0.1,
        ucb_alpha: float = 2.0,
        **model_params: Any,
    ) -> None:
        """Initialize River contextual bandit.

        Args:
            n_actions: Number of available actions.
            exploration_algorithm: Exploration strategy ('epsilon-greedy', 'ucb', 'thompson').
            learning_rate: Learning rate for the optimizer.
            epsilon: Exploration parameter for epsilon-greedy.
            ucb_alpha: Confidence parameter for UCB.
            **model_params: Additional model parameters.
        """
        super().__init__(n_actions, exploration_algorithm, **model_params)

        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.ucb_alpha = ucb_alpha

        self.models = [
            linear_model.LinearRegression(
                optimizer=optim.SGD(learning_rate),
                intercept_lr=learning_rate,
            )
            for _ in range(n_actions)
        ]

        self.action_counts = np.zeros(n_actions)
        self.action_values = np.zeros(n_actions)
        self.action_variances = np.ones(n_actions)
        self.timestep = 0

    def predict(
        self,
        context: NDArray[np.float64],
    ) -> Tuple[int, NDArray[np.float64]]:
        """Select an action using the specified exploration strategy.

        Args:
            context: Context feature vector of shape (n_features,).

        Returns:
            Tuple containing selected action index and predicted values.
        """
        context_dict = {f"x{i}": float(v) for i, v in enumerate(context)}

        predicted_values = np.array(
            [
                model.predict_one(context_dict)
                if hasattr(model, "_weights") and model._weights
                else 0.0
                for model in self.models
            ]
        )

        if self.exploration_algorithm == "epsilon-greedy":
            action = self._epsilon_greedy(predicted_values)
        elif self.exploration_algorithm == "ucb":
            action = self._ucb(predicted_values)
        elif self.exploration_algorithm == "thompson":
            action = self._thompson_sampling(predicted_values)
        else:
            action = int(np.argmax(predicted_values))

        return action, predicted_values

    def update(
        self,
        context: NDArray[np.float64],
        action: int,
        reward: float,
    ) -> None:
        """Update the model for the selected action.

        Args:
            context: Context feature vector that was used.
            action: Action that was taken.
            reward: Reward received for the action.
        """
        context_dict = {f"x{i}": float(v) for i, v in enumerate(context)}

        self.models[action].learn_one(context_dict, reward)

        self.action_counts[action] += 1
        delta = reward - self.action_values[action]
        self.action_values[action] += delta / self.action_counts[action]
        self.action_variances[action] += delta * (reward - self.action_values[action])

        self.timestep += 1
        self._is_fitted = True

    def reset(self) -> None:
        """Reset all models to initial state."""
        self.models = [
            linear_model.LinearRegression(
                optimizer=optim.SGD(self.learning_rate),
                intercept_lr=self.learning_rate,
            )
            for _ in range(self.n_actions)
        ]
        self.action_counts = np.zeros(self.n_actions)
        self.action_values = np.zeros(self.n_actions)
        self.action_variances = np.ones(self.n_actions)
        self.timestep = 0
        self._is_fitted = False

    def _epsilon_greedy(self, predicted_values: NDArray[np.float64]) -> int:
        """Epsilon-greedy exploration strategy.

        Args:
            predicted_values: Predicted values for all actions.

        Returns:
            Selected action index.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(predicted_values))

    def _ucb(self, predicted_values: NDArray[np.float64]) -> int:
        """Upper Confidence Bound exploration strategy.

        Args:
            predicted_values: Predicted values for all actions.

        Returns:
            Selected action index.
        """
        if self.timestep < self.n_actions:
            return self.timestep

        ucb_values = predicted_values + self.ucb_alpha * np.sqrt(
            np.log(self.timestep + 1) / (self.action_counts + 1e-5)
        )
        return int(np.argmax(ucb_values))

    def _thompson_sampling(self, predicted_values: NDArray[np.float64]) -> int:
        """Thompson sampling exploration strategy.

        Args:
            predicted_values: Predicted values for all actions.

        Returns:
            Selected action index.
        """
        sampled_values = np.zeros(self.n_actions)
        for i in range(self.n_actions):
            if self.action_counts[i] > 0:
                std = np.sqrt(self.action_variances[i] / (self.action_counts[i] + 1))
            else:
                std = 1.0
            sampled_values[i] = np.random.normal(self.action_values[i], std)
        return int(np.argmax(sampled_values))
