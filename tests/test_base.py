import numpy as np
from cb_comparison.data.dataset import DatasetConfig, MessageDataset

from core import Context, Feedback


def test_context_creation() -> None:
    context = Context(features=[1.0, 2.0, 3.0], user_id="test_user")
    assert len(context.features) == 3
    assert context.user_id == "test_user"
    assert isinstance(context.to_array(), np.ndarray)


def test_context_to_array() -> None:
    features = [1.0, 2.0, 3.0]
    context = Context(features=features)
    array = context.to_array()
    assert np.allclose(array, np.array(features))


def test_feedback_creation() -> None:
    context = Context(features=[1.0, 2.0])
    feedback = Feedback(reward=1.5, action=0, context=context)
    assert feedback.reward == 1.5
    assert feedback.action == 0
    assert feedback.context == context


def test_dataset_creation() -> None:
    config = DatasetConfig(n_features=5, n_actions=3, n_samples=100)
    dataset = MessageDataset(config)
    assert dataset.config.n_features == 5
    assert dataset.config.n_actions == 3
    assert len(dataset.contexts_cache) == 100


def test_dataset_get_context() -> None:
    config = DatasetConfig(n_features=5, n_actions=3, n_samples=10)
    dataset = MessageDataset(config)
    context = dataset.get_context(0)
    assert len(context.features) == 5
    assert isinstance(context, Context)


def test_dataset_get_reward() -> None:
    config = DatasetConfig(n_features=5, n_actions=3, n_samples=10)
    dataset = MessageDataset(config)
    context = dataset.get_context(0)
    reward = dataset.get_reward(context, 0)
    assert isinstance(reward, float)


def test_dataset_simulate_feedback() -> None:
    config = DatasetConfig(n_features=5, n_actions=3, n_samples=10)
    dataset = MessageDataset(config)
    context = dataset.get_context(0)
    feedback = dataset.simulate_feedback(context, 0)
    assert isinstance(feedback, Feedback)
    assert feedback.action == 0
    assert feedback.context == context
