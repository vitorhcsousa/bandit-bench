from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel

from core.base import Context, Feedback


class DatasetConfig(BaseModel):
    n_features: int = 10
    n_actions: int = 5
    n_samples: int = 10000
    noise_level: float = 0.1
    random_seed: int = 42


class MessageDataset:
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        np.random.seed(config.random_seed)
        self.action_weights = self._generate_action_weights()
        self.contexts_cache: list[Context] = []
        self.optimal_actions_cache: list[int] = []
        self._generate_data()

    def _generate_action_weights(self) -> np.ndarray:
        return np.random.randn(self.config.n_actions, self.config.n_features)

    def _generate_data(self) -> None:
        for i in range(self.config.n_samples):
            context_features = np.random.randn(self.config.n_features).tolist()
            context = Context(
                features=context_features,
                user_id=f"user_{i % 1000}",
                metadata={"index": i},
            )
            self.contexts_cache.append(context)

            optimal_action = self._get_optimal_action(context)
            self.optimal_actions_cache.append(optimal_action)

    def _get_optimal_action(self, context: Context) -> int:
        context_array = context.to_array()
        expected_rewards = self.action_weights @ context_array
        return int(np.argmax(expected_rewards))

    def get_context(self, index: int) -> Context:
        return self.contexts_cache[index % len(self.contexts_cache)]

    def get_reward(self, context: Context, action: int) -> float:
        context_array = context.to_array()
        expected_rewards = self.action_weights @ context_array
        base_reward = expected_rewards[action]
        noise = np.random.normal(0, self.config.noise_level)
        return float(base_reward + noise)

    def get_optimal_action(self, index: int) -> int:
        return self.optimal_actions_cache[index % len(self.optimal_actions_cache)]

    def simulate_feedback(self, context: Context, action: int) -> Feedback:
        reward = self.get_reward(context, action)
        return Feedback(
            reward=reward,
            action=action,
            context=context,
            metadata={"timestamp": np.random.randint(0, 1000000)},
        )


@dataclass
class MessageAction:
    """Represents a message action/variant.

    Attributes:
        action_id: Unique identifier for the action.
        message_type: Type/category of the message.
        tone: Tone of the message (formal, casual, friendly, etc.).
        length: Length category (short, medium, long).
    """

    action_id: int
    message_type: str
    tone: str
    length: str


class MessageFeedbackDataset:
    """Dataset generator for message-feedback contextual bandit experiments.

    Simulates a scenario where different message variants are sent to users
    and feedback (reward) is collected based on user context.

    Attributes:
        n_actions: Number of different message variants.
        context_dim: Dimensionality of user context features.
        actions: list of message action definitions.
        noise_level: Standard deviation of reward noise.
    """

    def __init__(
        self,
        n_actions: int = 5,
        context_dim: int = 10,
        noise_level: float = 0.1,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize the message feedback dataset.

        Args:
            n_actions: Number of message variants/actions.
            context_dim: Dimension of context feature vectors.
            noise_level: Standard deviation for reward noise.
            random_seed: Random seed for reproducibility.
        """
        self.n_actions = n_actions
        self.context_dim = context_dim
        self.noise_level = noise_level

        self.rng = np.random.RandomState(random_seed)

        self.actions = self._create_actions()
        self.action_weights = self._create_action_weights()

    def _create_actions(self) -> list[MessageAction]:
        """Create message action definitions.

        Returns:
            list of MessageAction objects.
        """
        message_types = ["promotional", "informational", "reminder", "survey", "update"]
        tones = ["formal", "casual", "friendly", "urgent", "neutral"]
        lengths = ["short", "medium", "long"]

        actions = []
        for i in range(self.n_actions):
            action = MessageAction(
                action_id=i,
                message_type=message_types[i % len(message_types)],
                tone=tones[i % len(tones)],
                length=lengths[i % len(lengths)],
            )
            actions.append(action)

        return actions

    def _create_action_weights(self) -> NDArray[np.float64]:
        """Create weight vectors for each action defining reward relationships.

        Returns:
            Array of shape (n_actions, context_dim) with action-specific weights.
        """
        weights = self.rng.randn(self.n_actions, self.context_dim)
        weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)
        return weights

    def generate_context(self, n_samples: int = 1) -> NDArray[np.float64]:
        """Generate random user context features.

        Context features represent user characteristics like demographics,
        past behavior, preferences, time of day, etc.

        Args:
            n_samples: Number of context samples to generate.

        Returns:
            Array of shape (n_samples, context_dim) with context features.
        """
        contexts = self.rng.randn(n_samples, self.context_dim)
        contexts = contexts / np.linalg.norm(contexts, axis=1, keepdims=True)
        return contexts

    def get_optimal_action(self, context: NDArray[np.float64]) -> int:
        """Determine the optimal action for a given context.

        Args:
            context: Context feature vector of shape (context_dim,).

        Returns:
            Index of the optimal action.
        """
        expected_rewards = self.action_weights @ context
        return int(np.argmax(expected_rewards))

    def get_reward(
        self,
        context: NDArray[np.float64],
        action: int,
        deterministic: bool = False,
    ) -> float:
        """Get reward for taking an action in a context.

        Reward simulates user feedback/engagement with the sent message.

        Args:
            context: Context feature vector of shape (context_dim,).
            action: Action/message variant that was selected.
            deterministic: If True, return expected reward without noise.

        Returns:
            Reward value (higher is better).
        """
        expected_reward = float(self.action_weights[action] @ context)

        if deterministic:
            return expected_reward

        noise = self.rng.normal(0, self.noise_level)
        return expected_reward + noise

    def generate_batch(
        self,
        batch_size: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
        """Generate a batch of contexts with optimal actions and rewards.

        Args:
            batch_size: Number of samples to generate.

        Returns:
            tuple containing:
                - contexts: Array of shape (batch_size, context_dim)
                - optimal_actions: Array of shape (batch_size,)
                - optimal_rewards: Array of shape (batch_size,)
        """
        contexts = self.generate_context(batch_size)
        optimal_actions = np.array([self.get_optimal_action(ctx) for ctx in contexts])
        optimal_rewards = np.array(
            [
                self.get_reward(contexts[i], optimal_actions[i], deterministic=True)
                for i in range(batch_size)
            ]
        )

        return contexts, optimal_actions, optimal_rewards

    def to_dataframe(
        self,
        contexts: NDArray[np.float64],
        actions: Optional[NDArray[np.int64]] = None,
        rewards: Optional[NDArray[np.float64]] = None,
    ) -> pd.DataFrame:
        """Convert contexts, actions, and rewards to a pandas DataFrame.

        Args:
            contexts: Context feature vectors.
            actions: Action indices (optional).
            rewards: Reward values (optional).

        Returns:
            DataFrame with context features and optional actions/rewards.
        """
        data = {f"context_{i}": contexts[:, i] for i in range(self.context_dim)}

        if actions is not None:
            data["action"] = actions

        if rewards is not None:
            data["reward"] = rewards

        return pd.DataFrame(data)

    def get_action_info(self) -> pd.DataFrame:
        """Get information about all available actions.

        Returns:
            DataFrame with action details.
        """
        return pd.DataFrame([vars(action) for action in self.actions])
