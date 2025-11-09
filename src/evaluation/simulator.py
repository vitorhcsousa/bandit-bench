from typing import Any

import numpy as np
from cb_comparison.data.dataset import MessageDataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from core import BaseContextualBandit
from evaluation.metrics import MetricsCalculator


class BanditSimulator:
    def __init__(self, dataset: MessageDataset, seed: int = 42) -> None:
        self.dataset = dataset
        self.console = Console()
        self.metrics_calculator = MetricsCalculator()
        np.random.seed(seed)

    def run_simulation(
        self,
        bandits: list[BaseContextualBandit],
        n_rounds: int,
        verbose: bool = True,
    ) -> dict[str, Any]:
        for bandit in bandits:
            self.metrics_calculator.initialize_bandit(bandit.get_name())

        if verbose:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Running simulations...", total=n_rounds)

                for round_idx in range(n_rounds):
                    context = self.dataset.get_context(round_idx)
                    optimal_action = self.dataset.get_optimal_action(round_idx)
                    optimal_reward = self.dataset.get_reward(context, optimal_action)

                    for bandit in bandits:
                        action = bandit.select_action(context)
                        feedback = self.dataset.simulate_feedback(context, action)
                        bandit.update(feedback)

                        self.metrics_calculator.update(
                            bandit.get_name(),
                            feedback.reward,
                            action,
                            optimal_reward,
                        )

                    progress.update(task, advance=1)
        else:
            for round_idx in range(n_rounds):
                context = self.dataset.get_context(round_idx)
                optimal_action = self.dataset.get_optimal_action(round_idx)
                optimal_reward = self.dataset.get_reward(context, optimal_action)

                for bandit in bandits:
                    action = bandit.select_action(context)
                    feedback = self.dataset.simulate_feedback(context, action)
                    bandit.update(feedback)

                    self.metrics_calculator.update(
                        bandit.get_name(),
                        feedback.reward,
                        action,
                        optimal_reward,
                    )

        return self.metrics_calculator.get_summary()

    def print_results(self) -> None:
        comparison = self.metrics_calculator.get_comparison_table()

        table = Table(title="Contextual Bandit Comparison Results", show_lines=True)
        table.add_column("Rank", justify="center", style="cyan")
        table.add_column("Algorithm", justify="left", style="magenta")
        table.add_column("Avg Reward", justify="right", style="green")
        table.add_column("Total Reward", justify="right", style="yellow")
        table.add_column("Final Regret", justify="right", style="red")
        table.add_column("Rounds", justify="right", style="blue")

        for idx, row in enumerate(comparison, 1):
            table.add_row(
                str(idx),
                row["Algorithm"],
                row["Avg Reward"],
                row["Total Reward"],
                row["Final Regret"],
                str(row["Rounds"]),
            )

        self.console.print(table)

    def get_metrics(self) -> MetricsCalculator:
        return self.metrics_calculator
