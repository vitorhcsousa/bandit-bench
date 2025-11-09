from __future__ import annotations

from typing import Literal

import numpy as np
from contextualbandits.online import (  # type: ignore[import-untyped]
    BootstrappedTS,
    BootstrappedUCB,
    EpsilonGreedy,
    SoftmaxExplorer,
)
from sklearn.linear_model import SGDClassifier  # type: ignore[import-untyped]

from cb_comparison.core.base import BaseContextualBandit, Context, Feedback


class ContextualBanditsLibBandit(BaseContextualBandit):
    def __init__(
        self,
        n_actions: int,
        n_features: int,
        exploration: Literal[
            "epsilon-greedy", "softmax", "bootstrap-ucb", "bootstrap-ts"
        ] = "epsilon-greedy",
        epsilon: float = 0.1,
        n_bootstraps: int = 5,
    ) -> None:
        super().__init__(n_actions, n_features)
        self.exploration = exploration
        self.epsilon = epsilon
        self.n_bootstraps = n_bootstraps

        self.model = self._initialize_model()
        self.history_contexts: list[np.ndarray] = []
        self.history_actions: list[int] = []
        self.history_rewards: list[float] = []

    def _initialize_model(
        self,
    ) -> EpsilonGreedy | SoftmaxExplorer | BootstrappedUCB | BootstrappedTS:
        base_estimator = SGDClassifier(loss="log_loss", warm_start=True, random_state=42)

        if self.exploration == "epsilon-greedy":
            return EpsilonGreedy(
                base_estimator,
                nchoices=self.n_actions,
                explore_prob=self.epsilon,
                random_state=42,
            )
        elif self.exploration == "softmax":
            return SoftmaxExplorer(
                base_estimator,
                nchoices=self.n_actions,
                random_state=42,
            )
        elif self.exploration == "bootstrap-ucb":
            return BootstrappedUCB(
                base_estimator,
                nchoices=self.n_actions,
                nsamples=self.n_bootstraps,
                random_state=42,
            )
        elif self.exploration == "bootstrap-ts":
            return BootstrappedTS(
                base_estimator,
                nchoices=self.n_actions,
                nsamples=self.n_bootstraps,
                random_state=42,
            )
        else:
            return EpsilonGreedy(
                base_estimator,
                nchoices=self.n_actions,
                explore_prob=self.epsilon,
                random_state=42,
            )

    def select_action(self, context: Context) -> int:
        context_array = context.to_array().reshape(1, -1)

        if len(self.history_contexts) < self.n_actions:
            return len(self.history_contexts)

        action = self.model.predict(context_array)
        return int(action[0])

    def update(self, feedback: Feedback) -> None:
        context_array = feedback.context.to_array()
        self.history_contexts.append(context_array)
        self.history_actions.append(feedback.action)
        self.history_rewards.append(feedback.reward)

        if len(self.history_contexts) >= self.n_actions:
            x = np.array(self.history_contexts)
            a = np.array(self.history_actions)
            r = np.array(self.history_rewards)

            self.model.fit(x, a, r)

        self.n_rounds += 1

    def get_name(self) -> str:
        return f"ContextualBandits-{self.exploration}"

    def reset(self) -> None:
        super().reset()
        self.model = self._initialize_model()
        self.history_contexts = []
        self.history_actions = []
        self.history_rewards = []
