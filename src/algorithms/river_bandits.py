from typing import Literal

import numpy as np
from river import linear_model, optim

from core.base import BaseContextualBandit, Context, Feedback


class RiverBandit(BaseContextualBandit):
    def __init__(
        self,
        n_actions: int,
        n_features: int,
        exploration: Literal["epsilon-greedy", "ucb", "softmax"] = "epsilon-greedy",
        epsilon: float = 0.1,
        temperature: float = 1.0,
        learning_rate: float = 0.01,
    ) -> None:
        super().__init__(n_actions, n_features)
        self.exploration = exploration
        self.epsilon = epsilon
        self.temperature = temperature
        self.learning_rate = learning_rate

        self.models = [
            linear_model.LinearRegression(optimizer=optim.SGD(learning_rate))
            for _ in range(n_actions)
        ]

        self.action_counts = np.zeros(n_actions)
        self.action_rewards = np.zeros(n_actions)

    def _context_to_dict(self, context: Context) -> dict[str, float]:
        return {f"f{i}": v for i, v in enumerate(context.features)}

    def select_action(self, context: Context) -> int:
        context_dict = self._context_to_dict(context)

        q_values = np.array([model.predict_one(context_dict) for model in self.models])

        if self.exploration == "epsilon-greedy":
            return self._epsilon_greedy(q_values)
        elif self.exploration == "ucb":
            return self._ucb(q_values)
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

    def _softmax(self, q_values: np.ndarray) -> int:
        exp_values = np.exp(q_values / self.temperature)
        probs = exp_values / np.sum(exp_values)
        return int(np.random.choice(self.n_actions, p=probs))

    def update(self, feedback: Feedback) -> None:
        context_dict = self._context_to_dict(feedback.context)

        self.models[feedback.action].learn_one(context_dict, feedback.reward)

        self.action_counts[feedback.action] += 1
        self.action_rewards[feedback.action] += feedback.reward
        self.n_rounds += 1

    def get_name(self) -> str:
        return f"River-{self.exploration}"

    def reset(self) -> None:
        super().reset()
        self.models = [
            linear_model.LinearRegression(optimizer=optim.SGD(self.learning_rate))
            for _ in range(self.n_actions)
        ]
        self.action_counts = np.zeros(self.n_actions)
        self.action_rewards = np.zeros(self.n_actions)
