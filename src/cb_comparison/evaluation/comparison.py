from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..bandits.base import ContextualBandit
from ..data.dataset import MessageFeedbackDataset
from .metrics import BanditMetrics


@dataclass
class ExperimentConfig:
    """Configuration for a bandit experiment.

    Attributes:
        name: Name of the experiment.
        bandit: Contextual bandit instance to evaluate.
        n_rounds: Number of rounds to run.
        batch_size: Number of interactions per round.
    """

    name: str
    bandit: ContextualBandit
    n_rounds: int = 1000
    batch_size: int = 1


class BanditComparison:
    """Framework for comparing multiple contextual bandit algorithms.

    Runs experiments across different bandit implementations and exploration
    strategies, collecting metrics for comparison.

    Attributes:
        dataset: Dataset generator for contexts and rewards.
        experiments: List of experiment configurations.
        results: Dictionary storing results for each experiment.
    """

    def __init__(
        self,
        dataset: MessageFeedbackDataset,
        random_seed: int = 42,
    ) -> None:
        """Initialize the comparison framework.

        Args:
            dataset: Dataset generator for the experiments.
            random_seed: Random seed for reproducibility.
        """
        self.dataset = dataset
        self.random_seed = random_seed
        self.experiments: List[ExperimentConfig] = []
        self.results: Dict[str, Dict[str, Any]] = {}

        np.random.seed(random_seed)

    def add_experiment(
        self,
        name: str,
        bandit: ContextualBandit,
        n_rounds: int = 1000,
    ) -> None:
        """Add an experiment configuration.

        Args:
            name: Unique name for the experiment.
            bandit: Contextual bandit instance to evaluate.
            n_rounds: Number of rounds to run.
        """
        config = ExperimentConfig(
            name=name,
            bandit=bandit,
            n_rounds=n_rounds,
        )
        self.experiments.append(config)

    def run_experiment(
        self,
        config: ExperimentConfig,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Run a single experiment.

        Args:
            config: Experiment configuration.
            show_progress: Whether to show progress bar.

        Returns:
            Dictionary containing experiment results and metrics.
        """
        metrics = BanditMetrics()
        bandit = config.bandit
        bandit.reset()

        iterator = tqdm(
            range(config.n_rounds),
            desc=f"Running {config.name}",
            disable=not show_progress,
        )

        for _ in iterator:
            context = self.dataset.generate_context(1)[0]

            action, scores = bandit.predict(context)

            reward = self.dataset.get_reward(context, action)

            optimal_action = self.dataset.get_optimal_action(context)
            optimal_reward = self.dataset.get_reward(
                context,
                optimal_action,
                deterministic=True,
            )

            bandit.update(context, action, reward)

            metrics.record_step(action, reward, optimal_reward)

        return {
            "config": config,
            "metrics": metrics,
            "bandit_info": bandit.get_info(),
        }

    def run_all(self, show_progress: bool = True) -> None:
        """Run all configured experiments.

        Args:
            show_progress: Whether to show progress bars.
        """
        for config in self.experiments:
            print(f"\n{'=' * 60}")
            print(f"Experiment: {config.name}")
            print(f"{'=' * 60}")

            result = self.run_experiment(config, show_progress)
            self.results[config.name] = result

            summary = result["metrics"].get_summary()
            print(f"\nResults:")
            print(f"  Total Reward: {summary['total_reward']:.2f}")
            print(f"  Average Regret: {summary['average_regret']:.4f}")
            print(f"  Cumulative Regret: {summary['cumulative_regret']:.2f}")

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Generate a comparison DataFrame of all experiments.

        Returns:
            DataFrame with comparative metrics for all experiments.
        """
        comparison_data = []

        for name, result in self.results.items():
            summary = result["metrics"].get_summary()
            bandit_info = result["bandit_info"]

            row = {
                "experiment": name,
                "library": result["config"].bandit.__class__.__name__,
                "exploration": bandit_info["exploration_algorithm"],
                "total_reward": summary["total_reward"],
                "mean_reward": summary["mean_reward"],
                "std_reward": summary["std_reward"],
                "cumulative_regret": summary["cumulative_regret"],
                "average_regret": summary["average_regret"],
                "n_rounds": summary["n_steps"],
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        df = df.sort_values("cumulative_regret")
        return df

    def get_regret_curves(self) -> pd.DataFrame:
        """Get cumulative regret curves for all experiments.

        Returns:
            DataFrame with regret curves over time for each experiment.
        """
        curves_data = []

        for name, result in self.results.items():
            metrics = result["metrics"]
            cumulative_regret = metrics.get_cumulative_regret()

            for t, regret in enumerate(cumulative_regret):
                curves_data.append(
                    {
                        "experiment": name,
                        "timestep": t,
                        "cumulative_regret": regret,
                    }
                )

        return pd.DataFrame(curves_data)

    def get_reward_curves(self, window: int = 100) -> pd.DataFrame:
        """Get moving average reward curves for all experiments.

        Args:
            window: Window size for moving average.

        Returns:
            DataFrame with reward curves over time for each experiment.
        """
        curves_data = []

        for name, result in self.results.items():
            metrics = result["metrics"]
            avg_rewards = metrics.get_average_reward(window)

            for t, reward in enumerate(avg_rewards):
                curves_data.append(
                    {
                        "experiment": name,
                        "timestep": t,
                        "average_reward": reward,
                    }
                )

        return pd.DataFrame(curves_data)
