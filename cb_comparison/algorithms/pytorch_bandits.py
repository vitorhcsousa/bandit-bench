from typing import Literal, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from cb_comparison.core.base import BaseContextualBandit, Context, Feedback


class LinearBanditModel(nn.Module):
    def __init__(self, n_features: int, n_actions: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, n_actions)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.linear(x))


class PyTorchLinearBandit(BaseContextualBandit):
    def __init__(
        self,
        n_actions: int,
        n_features: int,
        exploration: Literal["epsilon-greedy", "ucb", "thompson", "softmax"] = "epsilon-greedy",
        epsilon: float = 0.1,
        temperature: float = 1.0,
        learning_rate: float = 0.01,
    ) -> None:
        super().__init__(n_actions, n_features)
        self.exploration = exploration
        self.epsilon = epsilon
        self.temperature = temperature

        self.model = LinearBanditModel(n_features, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.action_counts = np.zeros(n_actions)
        self.action_rewards = np.zeros(n_actions)

    def select_action(self, context: Context) -> int:
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
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(q_values))

    def _ucb(self, q_values: np.ndarray) -> int:
        if self.n_rounds < self.n_actions:
            return self.n_rounds

        ucb_values = q_values + np.sqrt(2 * np.log(self.n_rounds + 1) / (self.action_counts + 1e-5))
        return int(np.argmax(ucb_values))

    def _thompson_sampling(self, q_values: np.ndarray) -> int:
        samples = np.random.normal(q_values, 1.0 / (self.action_counts + 1))
        return int(np.argmax(samples))

    def _softmax(self, q_values: np.ndarray) -> int:
        exp_values = np.exp(q_values / self.temperature)
        probs = exp_values / np.sum(exp_values)
        return int(np.random.choice(self.n_actions, p=probs))

    def update(self, feedback: Feedback) -> None:
        self.model.train()

        context_tensor = torch.FloatTensor(feedback.context.features).unsqueeze(0)
        target = torch.zeros(1, self.n_actions)

        with torch.no_grad():
            current_q = self.model(context_tensor)
            target = current_q.clone()

        target[0, feedback.action] = feedback.reward

        self.optimizer.zero_grad()
        output = self.model(context_tensor)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()

        self.action_counts[feedback.action] += 1
        self.action_rewards[feedback.action] += feedback.reward
        self.n_rounds += 1

    def get_name(self) -> str:
        return f"PyTorch-Linear-{self.exploration}"

    def reset(self) -> None:
        super().reset()
        self.model = LinearBanditModel(self.n_features, self.n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.action_counts = np.zeros(self.n_actions)
        self.action_rewards = np.zeros(self.n_actions)
