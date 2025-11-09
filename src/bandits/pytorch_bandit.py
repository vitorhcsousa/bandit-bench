from typing import Any, Optional

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.optim import Adam

from .base import ContextualBandit


class NeuralBanditModel(L.LightningModule):
    """Lightning neural network model for contextual bandits.

    A simple feedforward neural network that maps context features to
    action values for each arm.

    Attributes:
        layers: Sequential neural network layers.
        learning_rate: Learning rate for optimizer.
    """

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        hidden_dims: list[int],
        learning_rate: float = 0.001,
    ) -> None:
        """Initialize the neural bandit model.

        Args:
            input_dim: Dimension of input context features.
            n_actions: Number of actions/arms.
            hidden_dims: list of hidden layer dimensions.
            learning_rate: Learning rate for optimizer.
        """
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate

        layer_dims = [input_dim] + hidden_dims + [n_actions]
        layers = []

        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, n_actions).
        """
        return self.layers(x)

    def training_step(self, batch: tuple[torch.Tensor, int, float], batch_idx: int) -> torch.Tensor:
        """Lightning training step.

        Args:
            batch: tuple of (context, action, reward).
            batch_idx: Batch index.

        Returns:
            Loss tensor.
        """
        context, action, reward = batch
        q_values = self(context)
        predicted_value = q_values[0, action]
        target = torch.tensor([reward], dtype=torch.float32, device=self.device)

        loss = self.criterion(predicted_value, target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Adam:
        """Configure optimizer.

        Returns:
            Adam optimizer.
        """
        return Adam(self.parameters(), lr=self.learning_rate)


class PyTorchBandit(ContextualBandit):
    """Lightning-based contextual bandit with various exploration strategies.

    Implements contextual bandits using Lightning neural networks with support for
    multiple exploration algorithms including epsilon-greedy, UCB, and
    Thompson sampling.

    Attributes:
        model: Lightning neural network model.
        trainer: Lightning trainer for model updates.
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
        hidden_dims: Optional[list[int]] = None,
        learning_rate: float = 0.001,
        epsilon: float = 0.1,
        ucb_alpha: float = 2.0,
        device: Optional[str] = None,
        **model_params: Any,
    ) -> None:
        """Initialize Lightning PyTorch contextual bandit.

        Args:
            n_actions: Number of available actions.
            input_dim: Dimension of context features.
            exploration_algorithm: Exploration strategy ('epsilon-greedy', 'ucb', 'thompson').
            hidden_dims: list of hidden layer dimensions. Defaults to [64, 32].
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

        self.device_type = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = NeuralBanditModel(input_dim, n_actions, self.hidden_dims, learning_rate)

        self.trainer = L.Trainer(
            max_epochs=1,
            accelerator="auto",
            devices=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )

        self.action_counts = np.zeros(n_actions)
        self.action_values = np.zeros(n_actions)
        self.timestep = 0

    def predict(
        self,
        context: NDArray[np.float64],
    ) -> tuple[int, NDArray[np.float64]]:
        """Select an action using the specified exploration strategy.

        Args:
            context: Context feature vector of shape (n_features,).

        Returns:
            tuple containing selected action index and action scores.
        """
        self.model.eval()

        with torch.no_grad():
            context_tensor = torch.FloatTensor(context).unsqueeze(0)
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

        context_tensor = torch.FloatTensor(context).unsqueeze(0)
        batch = (context_tensor, action, reward)

        self.model.zero_grad()
        loss = self.model.training_step(batch, 0)
        loss.backward()

        optimizer = self.model.configure_optimizers()
        optimizer.step()

        self.action_counts[action] += 1
        self.action_values[action] += (reward - self.action_values[action]) / self.action_counts[
            action
        ]
        self.timestep += 1
        self._is_fitted = True

    def reset(self) -> None:
        """Reset the bandit to initial state."""
        self.model = NeuralBanditModel(
            self.input_dim, self.n_actions, self.hidden_dims, self.learning_rate
        )

        self.trainer = L.Trainer(
            max_epochs=1,
            accelerator="auto",
            devices=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )

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
