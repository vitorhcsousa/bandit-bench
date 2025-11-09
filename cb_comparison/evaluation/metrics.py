from typing import Any, cast

from pydantic import BaseModel, Field


class BanditMetrics(BaseModel):
    total_reward: float = 0.0
    cumulative_rewards: list[float] = Field(default_factory=list)
    cumulative_regret: list[float] = Field(default_factory=list)
    action_distribution: dict[int, int] = Field(default_factory=dict)
    average_reward: float = 0.0
    regret: float = 0.0
    n_rounds: int = 0

    model_config = {"arbitrary_types_allowed": True}


class MetricsCalculator:
    def __init__(self) -> None:
        self.metrics: dict[str, BanditMetrics] = {}

    def initialize_bandit(self, bandit_name: str) -> None:
        self.metrics[bandit_name] = BanditMetrics()

    def update(
        self,
        bandit_name: str,
        reward: float,
        action: int,
        optimal_reward: float,
    ) -> None:
        if bandit_name not in self.metrics:
            self.initialize_bandit(bandit_name)

        metrics = self.metrics[bandit_name]
        metrics.total_reward += reward
        metrics.n_rounds += 1

        cumulative_reward = (
            metrics.cumulative_rewards[-1] + reward if metrics.cumulative_rewards else reward
        )
        metrics.cumulative_rewards.append(cumulative_reward)

        instant_regret = optimal_reward - reward
        cumulative_regret = (
            metrics.cumulative_regret[-1] + instant_regret
            if metrics.cumulative_regret
            else instant_regret
        )
        metrics.cumulative_regret.append(cumulative_regret)

        metrics.action_distribution[action] = metrics.action_distribution.get(action, 0) + 1

        metrics.average_reward = metrics.total_reward / metrics.n_rounds
        metrics.regret = cumulative_regret

    def get_metrics(self, bandit_name: str) -> BanditMetrics:
        return self.metrics.get(bandit_name, BanditMetrics())

    def get_all_metrics(self) -> dict[str, BanditMetrics]:
        return self.metrics

    def get_summary(self) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = {}
        for name, metrics in self.metrics.items():
            summary[name] = {
                "average_reward": metrics.average_reward,
                "total_reward": metrics.total_reward,
                "final_regret": metrics.regret,
                "n_rounds": metrics.n_rounds,
                "action_distribution": metrics.action_distribution,
            }
        return summary

    def get_comparison_table(self) -> list[dict[str, Any]]:
        comparison: list[dict[str, Any]] = []
        for name, m in self.metrics.items():
            comparison.append(
                {
                    "Algorithm": name,
                    "Avg Reward": m.average_reward,
                    "Total Reward": m.total_reward,
                    "Final Regret": m.regret,
                    "Rounds": m.n_rounds,
                }
            )

        comparison.sort(key=lambda x: cast(float, x["Avg Reward"]), reverse=True)
        return comparison
