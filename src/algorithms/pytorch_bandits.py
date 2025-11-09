from typing import Literal, cast

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam

from core.base import BaseContextualBandit, Context, Feedback


class LinearBanditModel(L.LightningModule):
    """Lightning linear model for contextual bandits."""

    def __init__(self, n_features: int, n_actions: int, learning_rate: float = 0.01) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.linear = nn.Linear(n_features, n_actions)
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.linear(x))

    def training_step(self, batch: tuple[Tensor, int, float], batch_idx: int) -> Tensor:
        """Lightning training step.

        Args:
            batch: Tuple of (context, action, reward).
            batch_idx: Batch index.

        Returns:
            Loss tensor.
        """
        context, action, reward = batch

        q_values = self(context)

        with torch.no_grad():
            target = q_values.clone()
        target[0, action] = reward

        output = self(context)
        loss = self.criterion(output, target)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Adam:
        """Configure optimizer."""
        return Adam(self.parameters(), lr=self.learning_rate)


class PyTorchLinearBandit(BaseContextualBandit):
    """Lightning-based linear contextual bandit.

    Uses PyTorch Lightning for training and model management.

    Attributes:
        model: Lightning linear model.
        exploration: Exploration strategy.
        epsilon: Exploration parameter for epsilon-greedy.
        temperature: Temperature parameter for softmax.
        action_counts: Count of times each action was selected.
        action_rewards: Sum of rewards for each action.
    """

    def __init__(
        self,
        n_actions: int,
        n_features: int,
        exploration: Literal["epsilon-greedy", "ucb", "thompson", "softmax"] = "epsilon-greedy",
        epsilon: float = 0.1,
        temperature: float = 1.0,
        learning_rate: float = 0.01,
    ) -> None:
        """Initialize Lightning linear bandit.

        Args:
            n_actions: Number of available actions.
            n_features: Dimension of context features.
            exploration: Exploration strategy.
            epsilon: Exploration parameter for epsilon-greedy.
            temperature: Temperature parameter for softmax.
            learning_rate: Learning rate for optimizer.
        """
        super().__init__(n_actions, n_features)
        self.exploration = exploration
        self.epsilon = epsilon
        self.temperature = temperature
        self.learning_rate = learning_rate

        self.model = LinearBanditModel(n_features, n_actions, learning_rate)

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
        self.action_rewards = np.zeros(n_actions)

    def select_action(self, context: Context) -> int:
        """Select an action using the specified exploration strategy.

        Args:
            context: Context features.

        Returns:
            Selected action index.
        """
        self.model.eval()
        with torch.no_grad():
            context_tensor = torch.FloatTensor(context.features).unsqueeze(0)
            q_values = self.model(context_tensor).squeeze().numpy()

        if self.exploration == "epsilon-greedy":
            return self._epsilon_greedy(q_values)
        elif self.exploration == "ucb":
            return self._ucb(q_values)
        elif self.exploration == "thompson":
            return self._thompson_sampling(q_values)
        elif self.exploration == "softmax":
            return self._softmax(q_values)
        else:
            return int(np.argmax(q_values))

    def _epsilon_greedy(self, q_values: np.ndarray) -> int:
        """Epsilon-greedy exploration."""
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(q_values))

    def _ucb(self, q_values: np.ndarray) -> int:
        """Upper Confidence Bound exploration."""
        if self.n_rounds < self.n_actions:
            return self.n_rounds

        ucb_values = q_values + np.sqrt(2 * np.log(self.n_rounds + 1) / (self.action_counts + 1e-5))
        return int(np.argmax(ucb_values))

    def _thompson_sampling(self, q_values: np.ndarray) -> int:
        """Thompson sampling exploration."""
        samples = np.random.normal(q_values, 1.0 / (self.action_counts + 1))
        return int(np.argmax(samples))

    def _softmax(self, q_values: np.ndarray) -> int:
        """Softmax exploration."""
        exp_values = np.exp(q_values / self.temperature)
        probs = exp_values / np.sum(exp_values)
        return int(np.random.choice(self.n_actions, p=probs))

    def update(self, feedback: Feedback) -> None:
        """Update the model with observed reward.

        Args:
            feedback: Feedback containing context, action, and reward.
        """
        self.model.train()

        context_tensor = torch.FloatTensor(feedback.context.features).unsqueeze(0)
        batch = (context_tensor, feedback.action, feedback.reward)

        self.model.zero_grad()
        loss = self.model.training_step(batch, 0)
        loss.backward()

        optimizer = self.model.configure_optimizers()
        optimizer.step()

        self.action_counts[feedback.action] += 1
        self.action_rewards[feedback.action] += feedback.reward
        self.n_rounds += 1

    def get_name(self) -> str:
        """Get bandit name."""
        return f"PyTorch-Linear-Lightning-{self.exploration}"

    def reset(self) -> None:
        """Reset the bandit to initial state."""
        super().reset()
        self.model = LinearBanditModel(self.n_features, self.n_actions, self.learning_rate)

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
        self.action_rewards = np.zeros(self.n_actions)
