from __future__ import annotations

from typing import Any, Literal

import numpy as np
from vowpalwabbit import pyvw  # type: ignore[import-untyped]

from core.base import BaseContextualBandit, Context, Feedback


class VowpalWabbitBandit(BaseContextualBandit):
    def __init__(
        self,
        n_actions: int,
        n_features: int,
        exploration: Literal["epsilon-greedy", "bag", "cover", "softmax"] = "epsilon-greedy",
        epsilon: float = 0.1,
        bag_size: int = 5,
        cover_size: int = 5,
        lambda_param: float = 1.0,
    ) -> None:
        super().__init__(n_actions, n_features)
        self.exploration = exploration
        self.epsilon = epsilon
        self.bag_size = bag_size
        self.cover_size = cover_size
        self.lambda_param = lambda_param

        self.vw = self._initialize_vw()

    def _initialize_vw(self) -> Any:
        if self.exploration == "epsilon-greedy":
            args = f"--cb_explore {self.n_actions} --epsilon {self.epsilon}"
        elif self.exploration == "bag":
            args = f"--cb_explore_adf --bag {self.bag_size}"
        elif self.exploration == "cover":
            args = f"--cb_explore_adf --cover {self.cover_size}"
        elif self.exploration == "softmax":
            args = f"--cb_explore {self.n_actions} --softmax --lambda {self.lambda_param}"
        else:
            args = f"--cb_explore {self.n_actions}"

        return pyvw.Workspace(args + " --quiet")

    def _context_to_vw_format(self, context: Context, action: int | None = None) -> str:
        features_str = " ".join([f"{i}:{v:.6f}" for i, v in enumerate(context.features)])

        if action is None:
            return f"| {features_str}"
        return f"{action}:0:1.0 | {features_str}"

    def select_action(self, context: Context) -> int:
        if self.exploration in ["bag", "cover"]:
            return self._select_action_adf(context)

        vw_example = self._context_to_vw_format(context)
        pmf = self.vw.predict(vw_example)

        return int(np.argmax(pmf)) if isinstance(pmf, (list, np.ndarray)) else int(pmf) - 1

    def _select_action_adf(self, context: Context) -> int:
        features_str = " ".join([f"{i}:{v:.6f}" for i, v in enumerate(context.features)])

        examples = [f"shared | {features_str}"]
        for action in range(self.n_actions):
            examples.append(f"| action_{action}")

        multiline_example = "\n".join(examples)
        pmf = self.vw.predict(multiline_example)

        return int(np.argmax(pmf)) if isinstance(pmf, (list, np.ndarray)) else int(pmf)

    def update(self, feedback: Feedback) -> None:
        if self.exploration in ["bag", "cover"]:
            self._update_adf(feedback)
        else:
            cost = -feedback.reward
            vw_example = f"{feedback.action + 1}:{cost}:1.0 | " + " ".join(
                [f"{i}:{v:.6f}" for i, v in enumerate(feedback.context.features)]
            )
            self.vw.learn(vw_example)

        self.n_rounds += 1

    def _update_adf(self, feedback: Feedback) -> None:
        features_str = " ".join([f"{i}:{v:.6f}" for i, v in enumerate(feedback.context.features)])

        cost = -feedback.reward
        examples = [f"shared | {features_str}"]
        for action in range(self.n_actions):
            if action == feedback.action:
                examples.append(f"{action}:{cost}:1.0 | action_{action}")
            else:
                examples.append(f"| action_{action}")

        multiline_example = "\n".join(examples)
        self.vw.learn(multiline_example)

    def get_name(self) -> str:
        return f"VowpalWabbit-{self.exploration}"

    def reset(self) -> None:
        super().reset()
        self.vw.finish()
        self.vw = self._initialize_vw()

    def __del__(self) -> None:
        if hasattr(self, "vw"):
            self.vw.finish()
