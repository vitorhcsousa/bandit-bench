from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray

from .base import ContextualBandit


class NeuralBanditModel(nn.Module):
    """Neural network model for contextual bandits.

    A simple feedforward neural network that maps context features to
    action values for each arm.

    Attributes:
        layers: Sequential neural network layers.
    """

    def __init__(self, input_dim: int, n_actions: int, hidden_dims: List[int]) -> None:
        """Initialize the neural bandit model.

        Args:
            input_dim: Dimension of input context features.
            n_actions: Number of actions/arms.
            hidden_dims: List of hidden layer dimensions.
        """
        super().__init__()

        layer_dims = [input_dim] + hidden_dims + [n_actions]
        layers = []

        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, n_actions).
        """
        return self.layers(x)


class PyTorchBandit(ContextualBandit):
    """PyTorch-based contextual bandit with various exploration strategies.

    Implements contextual bandits using neural networks with support for
    multiple exploration algorithms including epsilon-greedy, UCB, and
    Thompson sampling.

    Attributes:
        model: Neural network model.
        optimizer: PyTorch optimizer.
        criterion: Loss function.
        device: Device for computation (CPU/GPU).
        epsilon: Exploration parameter for epsilon-greedy.
        ucb_alpha: Confidence parameter for UCB.
        action_counts: Count of times each action was selected.
        action_values: Running average of rewards per action.
    """

    def __init__(
        self,
        n_actions: int,
        input_dim: int,
        exploration_algorithm: str = "epsilon-greedy",
        hidden_dims: Optional[List[int]] = None,
        learning_rate: float = 0.001,
        epsilon: float = 0.1,
        ucb_alpha: float = 2.0,
        device: Optional[str] = None,
        **model_params: Any,
    ) -> None:
        """Initialize PyTorch contextual bandit.

        Args:
            n_actions: Number of available actions.
            input_dim: Dimension of context features.
            exploration_algorithm: Exploration strategy ('epsilon-greedy', 'ucb', 'thompson').
            hidden_dims: List of hidden layer dimensions. Defaults to [64, 32].
            learning_rate: Learning rate for optimizer.
            epsilon: Exploration parameter for epsilon-greedy.
            ucb_alpha: Confidence parameter for UCB.
            device: Device for computation. Defaults to 'cuda' if available, else 'cpu'.
            **model_params: Additional model parameters.
        """
        super().__init__(n_actions, exploration_algorithm, **model_params)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.ucb_alpha = ucb_alpha

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = NeuralBanditModel(input_dim, n_actions, self.hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.action_counts = np.zeros(n_actions)
        self.action_values = np.zeros(n_actions)
        self.timestep = 0

    def predict(
        self,
        context: NDArray[np.float64],
    ) -> Tuple[int, NDArray[np.float64]]:
        """Select an action using the specified exploration strategy.

        Args:
            context: Context feature vector of shape (n_features,).

        Returns:
            Tuple containing selected action index and action scores.
        """
        self.model.eval()

        with torch.no_grad():
            context_tensor = torch.FloatTensor(context).unsqueeze(0).to(self.device)
            q_values = self.model(context_tensor).cpu().numpy().flatten()

        if self.exploration_algorithm == "epsilon-greedy":
            action = self._epsilon_greedy(q_values)
        elif self.exploration_algorithm == "ucb":
            action = self._ucb(q_values)
        elif self.exploration_algorithm == "thompson":
            action = self._thompson_sampling(q_values)
        else:
            action = int(np.argmax(q_values))

        return action, q_values

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
        self.model.train()

        context_tensor = torch.FloatTensor(context).unsqueeze(0).to(self.device)
        target = torch.FloatTensor([reward]).to(self.device)

        self.optimizer.zero_grad()
        q_values = self.model(context_tensor)
        predicted_value = q_values[0, action]

        loss = self.criterion(predicted_value, target)
        loss.backward()
        self.optimizer.step()

        self.action_counts[action] += 1
        self.action_values[action] += (reward - self.action_values[action]) / self.action_counts[
            action
        ]
        self.timestep += 1
        self._is_fitted = True

    def reset(self) -> None:
        """Reset the bandit to initial state."""
        self.model = NeuralBanditModel(self.input_dim, self.n_actions, self.hidden_dims).to(
            self.device
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.action_counts = np.zeros(self.n_actions)
        self.action_values = np.zeros(self.n_actions)
        self.timestep = 0
        self._is_fitted = False

    def _epsilon_greedy(self, q_values: NDArray[np.float64]) -> int:
        """Epsilon-greedy exploration strategy.

        Args:
            q_values: Q-values for all actions.

        Returns:
            Selected action index.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(q_values))

    def _ucb(self, q_values: NDArray[np.float64]) -> int:
        """Upper Confidence Bound exploration strategy.

        Args:
            q_values: Q-values for all actions.

        Returns:
            Selected action index.
        """
        if self.timestep < self.n_actions:
            return self.timestep

        ucb_values = q_values + self.ucb_alpha * np.sqrt(
            np.log(self.timestep + 1) / (self.action_counts + 1e-5)
        )
        return int(np.argmax(ucb_values))

    def _thompson_sampling(self, q_values: NDArray[np.float64]) -> int:
        """Thompson sampling exploration strategy.

        Args:
            q_values: Q-values for all actions.

        Returns:
            Selected action index.
        """
        sampled_values = np.zeros(self.n_actions)
        for i in range(self.n_actions):
            mean = self.action_values[i] if self.action_counts[i] > 0 else 0.5
            std = 1.0 / np.sqrt(self.action_counts[i] + 1)
            sampled_values[i] = np.random.normal(mean, std)
        return int(np.argmax(sampled_values))
