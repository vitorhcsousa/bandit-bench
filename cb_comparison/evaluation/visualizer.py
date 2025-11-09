from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]

from cb_comparison.evaluation.metrics import MetricsCalculator


class ResultsVisualizer:
    def __init__(self, metrics_calculator: MetricsCalculator) -> None:
        self.metrics_calculator = metrics_calculator
        sns.set_theme(style="whitegrid")
        self.colors = sns.color_palette("husl", len(metrics_calculator.metrics))

    def plot_cumulative_rewards(self, save_path: Path | None = None) -> None:
        plt.figure(figsize=(12, 6))

        for idx, (name, metrics) in enumerate(self.metrics_calculator.metrics.items()):
            plt.plot(
                metrics.cumulative_rewards,
                label=name,
                color=self.colors[idx],
                linewidth=2,
            )

        plt.xlabel("Round", fontsize=12)
        plt.ylabel("Cumulative Reward", fontsize=12)
        plt.title("Cumulative Rewards Over Time", fontsize=14, fontweight="bold")
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_cumulative_regret(self, save_path: Path | None = None) -> None:
        plt.figure(figsize=(12, 6))

        for idx, (name, metrics) in enumerate(self.metrics_calculator.metrics.items()):
            plt.plot(
                metrics.cumulative_regret,
                label=name,
                color=self.colors[idx],
                linewidth=2,
            )

        plt.xlabel("Round", fontsize=12)
        plt.ylabel("Cumulative Regret", fontsize=12)
        plt.title("Cumulative Regret Over Time", fontsize=14, fontweight="bold")
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_action_distribution(self, save_path: Path | None = None) -> None:
        n_bandits = len(self.metrics_calculator.metrics)
        fig, axes = plt.subplots(1, n_bandits, figsize=(5 * n_bandits, 5))

        # `axes` pode ser um único Axes ou uma lista/array de Axes: normalizar para list[Axes]
        if n_bandits == 1:
            axes_list: list[plt.Axes] = [cast(plt.Axes, axes)]
        else:
            axes_list = cast(list[plt.Axes], axes)

        for idx, (name, metrics) in enumerate(self.metrics_calculator.metrics.items()):
            actions = list(metrics.action_distribution.keys())
            counts = list(metrics.action_distribution.values())

            axes_list[idx].bar(actions, counts, color=self.colors[idx], alpha=0.7)
            axes_list[idx].set_xlabel("Action", fontsize=10)
            axes_list[idx].set_ylabel("Count", fontsize=10)
            axes_list[idx].set_title(name, fontsize=11, fontweight="bold")
            axes_list[idx].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_average_rewards_comparison(self, save_path: Path | None = None) -> None:
        names = list(self.metrics_calculator.metrics.keys())
        avg_rewards = [
            metrics.average_reward for metrics in self.metrics_calculator.metrics.values()
        ]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(names)), avg_rewards, color=self.colors, alpha=0.7)

        plt.xlabel("Algorithm", fontsize=12)
        plt.ylabel("Average Reward", fontsize=12)
        plt.title("Average Reward Comparison", fontsize=14, fontweight="bold")
        plt.xticks(range(len(names)), names, rotation=45, ha="right")
        plt.grid(True, alpha=0.3, axis="y")

        for bar, reward in zip(bars, avg_rewards):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{reward:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def create_all_plots(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        self.plot_cumulative_rewards(output_dir / "cumulative_rewards.png")
        self.plot_cumulative_regret(output_dir / "cumulative_regret.png")
        self.plot_action_distribution(output_dir / "action_distribution.png")
        self.plot_average_rewards_comparison(output_dir / "average_rewards.png")

    def export_results_to_csv(self, output_path: Path) -> None:
        comparison = self.metrics_calculator.get_comparison_table()
        df = pd.DataFrame(comparison)
        df.to_csv(output_path, index=False)
