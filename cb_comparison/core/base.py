from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel


class Context(BaseModel):
    features: list[float]
    user_id: str | None = None
    metadata: dict[str, Any] | None = None

    def to_array(self) -> np.ndarray:
        return np.array(self.features)


class Feedback(BaseModel):
    reward: float
    action: int
    context: Context
    metadata: dict[str, Any] | None = None


class BaseContextualBandit(ABC):
    def __init__(self, n_actions: int, n_features: int) -> None:
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_rounds = 0

    @abstractmethod
    def select_action(self, context: Context) -> int:
        pass

    @abstractmethod
    def update(self, feedback: Feedback) -> None:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    def reset(self) -> None:
        self.n_rounds = 0
