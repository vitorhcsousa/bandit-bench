from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray
from vowpalwabbit import pyvw

from .base import ContextualBandit


class VowpalWabbitBandit(ContextualBandit):
    """Vowpal Wabbit contextual bandit implementation.

    Wraps Vowpal Wabbit's contextual bandit algorithms including epsilon-greedy,
    bag, cover, and regcb exploration strategies.

    Attributes:
        vw: Vowpal Wabbit workspace instance.
        vw_args: Arguments passed to VW initialization.
    """

    def __init__(
        self,
        n_actions: int,
        exploration_algorithm: str = "epsilon-greedy",
        epsilon: float = 0.1,
        bag: int = 5,
        cover: int = 5,
        learning_rate: float = 0.1,
        **model_params: Any,
    ) -> None:
        """Initialize Vowpal Wabbit contextual bandit.

        Args:
            n_actions: Number of available actions.
            exploration_algorithm: VW exploration strategy ('epsilon-greedy', 'bag', 'cover', 'regcb').
            epsilon: Exploration parameter for epsilon-greedy.
            bag: Number of policies for bagging.
            cover: Number of policies for cover.
            learning_rate: Learning rate for updates.
            **model_params: Additional VW parameters.
        """
        super().__init__(n_actions, exploration_algorithm, **model_params)

        self.epsilon = epsilon
        self.bag = bag
        self.cover = cover
        self.learning_rate = learning_rate

        self.vw_args = self._build_vw_args()
        self.vw = pyvw.Workspace(self.vw_args)

    def _build_vw_args(self) -> str:
        """Build Vowpal Wabbit argument string.

        Returns:
            VW argument string.
        """
        if self.exploration_algorithm == "epsilon-greedy":
            return f"--cb {self.n_actions} --quiet -l {self.learning_rate} --cb_explore {self.n_actions} --epsilon {self.epsilon}"
        elif self.exploration_algorithm == "bag":
            return f"--cb_explore_adf --bag {self.bag} --quiet -l {self.learning_rate}"
        elif self.exploration_algorithm == "cover":
            return f"--cb_explore_adf --cover {self.cover} --quiet -l {self.learning_rate}"
        elif self.exploration_algorithm == "regcb":
            return f"--cb_explore_adf --regcb --quiet -l {self.learning_rate}"
        else:
            return f"--cb {self.n_actions} --quiet -l {self.learning_rate} --cb_explore {self.n_actions} --epsilon {self.epsilon}"

    def predict(
        self,
        context: NDArray[np.float64],
    ) -> Tuple[int, NDArray[np.float64]]:
        """Select an action using VW's exploration strategy.

        Args:
            context: Context feature vector of shape (n_features,).

        Returns:
            Tuple containing selected action index and action probabilities.
        """
        if self.exploration_algorithm in ["bag", "cover", "regcb"]:
            return self._predict_adf(context)
        else:
            return self._predict_simple(context)

    def _predict_simple(
        self,
        context: NDArray[np.float64],
    ) -> Tuple[int, NDArray[np.float64]]:
        """Predict using simple CB format.

        Args:
            context: Context feature vector.

        Returns:
            Tuple containing selected action and action probabilities.
        """
        context_str = " ".join([f"{i}:{v}" for i, v in enumerate(context) if v != 0])
        vw_example = f"| {context_str}"

        prediction = self.vw.predict(vw_example)

        # With --cb_explore, VW returns a list of probabilities
        if isinstance(prediction, list):
            probs = np.array(prediction)
            # Normalize to ensure probabilities sum to 1.0 (handle floating point errors)
            probs = probs / probs.sum()
            # Sample action based on probabilities
            action = np.random.choice(self.n_actions, p=probs)
        else:
            # Fallback for simple --cb (returns action directly)
            action = int(prediction) - 1
            probs = np.zeros(self.n_actions)
            probs[action] = 1.0

        return int(action), probs

    def _predict_adf(
        self,
        context: NDArray[np.float64],
    ) -> Tuple[int, NDArray[np.float64]]:
        """Predict using ADF (Action Dependent Features) format.

        Args:
            context: Context feature vector.

        Returns:
            Tuple containing selected action and action probabilities.
        """
        context_str = " ".join([f"{i}:{v}" for i, v in enumerate(context) if v != 0])

        examples = [f"shared | {context_str}"]
        for action in range(self.n_actions):
            examples.append(f"| action_{action}")

        predictions = self.vw.predict(examples)

        # VW ADF returns a list of (action_index, probability) tuples
        # or sometimes just a list of probabilities
        if isinstance(predictions, list) and len(predictions) > 0:
            if isinstance(predictions[0], tuple):
                # Format: [(action, prob), ...]
                # Sort by probability and take the most likely
                sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
                probs = np.zeros(self.n_actions)
                for action_idx, prob in predictions:
                    probs[action_idx] = prob
                # Normalize probabilities
                probs = probs / probs.sum()
                # Sample based on probabilities
                action = np.random.choice(self.n_actions, p=probs)
            else:
                # Format: [prob1, prob2, ...]
                probs = np.array(predictions)
                # Normalize to ensure sum is 1.0
                probs = probs / probs.sum()
                action = np.random.choice(self.n_actions, p=probs)
        else:
            # Fallback: uniform random
            action = np.random.randint(self.n_actions)
            probs = np.ones(self.n_actions) / self.n_actions

        return int(action), probs

    def update(
        self,
        context: NDArray[np.float64],
        action: int,
        reward: float,
    ) -> None:
        """Update VW model with observed reward.

        Args:
            context: Context feature vector that was used.
            action: Action that was taken.
            reward: Reward received for the action.
        """
        if self.exploration_algorithm in ["bag", "cover", "regcb"]:
            self._update_adf(context, action, reward)
        else:
            self._update_simple(context, action, reward)

        self._is_fitted = True

    def _update_simple(
        self,
        context: NDArray[np.float64],
        action: int,
        reward: float,
    ) -> None:
        """Update using simple CB format.

        Args:
            context: Context feature vector.
            action: Action taken.
            reward: Reward received.
        """
        context_str = " ".join([f"{i}:{v}" for i, v in enumerate(context) if v != 0])
        cost = -reward
        vw_example = f"{action + 1}:{cost}:1.0 | {context_str}"

        self.vw.learn(vw_example)

    def _update_adf(
        self,
        context: NDArray[np.float64],
        action: int,
        reward: float,
    ) -> None:
        """Update using ADF format.

        Args:
            context: Context feature vector.
            action: Action taken.
            reward: Reward received.
        """
        context_str = " ".join([f"{i}:{v}" for i, v in enumerate(context) if v != 0])
        cost = -reward

        examples = [f"shared | {context_str}"]
        for a in range(self.n_actions):
            if a == action:
                examples.append(f"0:{cost}:1.0 | action_{a}")
            else:
                examples.append(f"| action_{a}")

        self.vw.learn(examples)

    def reset(self) -> None:
        """Reset the VW workspace to initial state."""
        self.vw.finish()
        self.vw = pyvw.Workspace(self.vw_args)
        self._is_fitted = False

    def __del__(self) -> None:
        """Cleanup VW workspace."""
        if hasattr(self, "vw"):
            self.vw.finish()
