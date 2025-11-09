import numpy as np
from pydantic import BaseModel

from cb_comparison.core.base import Context, Feedback


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
