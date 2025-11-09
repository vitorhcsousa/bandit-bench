from typing import Any, Optional, Tuple

import numpy as np
from contextualbandits.online import (
    BootstrappedTS,
    EpsilonGreedy,
    LinUCB,
    SoftmaxExplorer,
)
from numpy.typing import NDArray
from sklearn.linear_model import SGDClassifier

from .base import ContextualBandit


class ContextualBanditsWrapper(ContextualBandit):
    """Wrapper for contextualbandits library implementations.

    Provides a unified interface to various algorithms from the contextualbandits
    library including LinUCB, Thompson Sampling, and Epsilon-Greedy.

    Attributes:
        model: The underlying contextualbandits model.
        base_algorithm: Base learning algorithm used.
    """

    def __init__(
        self,
        n_actions: int,
        exploration_algorithm: str = "linucb",
        alpha: float = 1.0,
        epsilon: float = 0.1,
        beta_prior: Optional[Tuple[Tuple[float, float], int]] = None,
        n_samples: int = 10,
        **model_params: Any,
    ) -> None:
        """Initialize contextualbandits wrapper.

        Args:
            n_actions: Number of available actions.
            exploration_algorithm: Algorithm to use ('linucb', 'thompson', 'epsilon-greedy', 'softmax').
            alpha: Confidence parameter for LinUCB.
            epsilon: Exploration parameter for epsilon-greedy.
            beta_prior: Prior parameters for Thompson Sampling in format ((alpha, beta), n).
                       If None, uses ((1.0, 1.0), 2) as default.
            n_samples: Number of bootstrap samples for Thompson Sampling.
            **model_params: Additional model parameters.
        """
        super().__init__(n_actions, exploration_algorithm, **model_params)

        self.alpha = alpha
        self.epsilon = epsilon
        self.beta_prior = beta_prior if beta_prior is not None else ((1.0, 1.0), 2)
        self.n_samples = n_samples

        self.base_algorithm = SGDClassifier(
            loss="log_loss",
            random_state=42,
        )
        self.model = self._create_model()

        self._first_update = [True] * n_actions

    def _create_model(self) -> Any:
        """Create the appropriate contextualbandits model.

        Returns:
            Initialized contextualbandits model.
        """
        base_alg = self.base_algorithm

        if self.exploration_algorithm == "linucb":
            return LinUCB(
                nchoices=self.n_actions,
                alpha=self.alpha,
            )
        elif self.exploration_algorithm == "thompson":
            return BootstrappedTS(
                base_algorithm=base_alg,
                nchoices=self.n_actions,
                nsamples=self.n_samples,
                beta_prior=self.beta_prior,
                batch_train=True,
            )
        elif self.exploration_algorithm == "epsilon-greedy":
            return EpsilonGreedy(
                base_algorithm=base_alg,
                nchoices=self.n_actions,
                explore_prob=self.epsilon,
                batch_train=True,
            )
        elif self.exploration_algorithm == "softmax":
            return SoftmaxExplorer(
                base_algorithm=base_alg,
                nchoices=self.n_actions,
                batch_train=True,
            )
        else:
            return LinUCB(nchoices=self.n_actions, alpha=self.alpha)

    def predict(
        self,
        context: NDArray[np.float64],
    ) -> Tuple[int, NDArray[np.float64]]:
        """Select an action using the model's policy.

        Args:
            context: Context feature vector of shape (n_features,).

        Returns:
            Tuple containing selected action index and action scores.
        """
        context_2d = context.reshape(1, -1)

        action = self.model.predict(context_2d)

        scores = np.zeros(self.n_actions)
        scores[action] = 1.0

        return int(action), scores

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
        context_2d = context.reshape(1, -1)

        if self.exploration_algorithm in {"thompson", "epsilon-greedy", "softmax"}:
            r_to_use = 1.0 if reward > 0.0 else 0.0
        else:
            r_to_use = reward
        self.model.partial_fit(
            context_2d,
            np.array([action]),
            np.array([r_to_use]),
        )
        self._first_update[action] = False
        self._is_fitted = True

    def reset(self) -> None:
        """Reset the model to initial state."""
        self.model = self._create_model()
        self._first_update = [True] * self.n_actions
        self._is_fitted = False
