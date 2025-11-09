from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots


class BanditVisualizer:
    """Visualization tools for contextual bandit comparison results.

    Provides methods to create various plots for analyzing and comparing
    bandit algorithm performance.

    Attributes:
        style: Matplotlib/Seaborn style to use.
        color_palette: Color palette for plots.
    """

    def __init__(
        self,
        style: str = "seaborn-v0_8-darkgrid",
        color_palette: str = "husl",
    ) -> None:
        """Initialize the visualizer.

        Args:
            style: Matplotlib style to use.
            color_palette: Seaborn color palette name.
        """
        self.style = style
        self.color_palette = color_palette

        plt.style.use(style)
        sns.set_palette(color_palette)

    def plot_regret_curves(
        self,
        regret_df: pd.DataFrame,
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot cumulative regret curves for all experiments.

        Args:
            regret_df: DataFrame with regret curves.
            save_path: Optional path to save the plot.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for experiment in regret_df["experiment"].unique():
            exp_data = regret_df[regret_df["experiment"] == experiment]
            ax.plot(
                exp_data["timestep"],
                exp_data["cumulative_regret"],
                label=experiment,
                linewidth=2,
            )

        ax.set_xlabel("Timestep", fontsize=12)
        ax.set_ylabel("Cumulative Regret", fontsize=12)
        ax.set_title("Cumulative Regret Comparison", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_reward_curves(
        self,
        reward_df: pd.DataFrame,
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot moving average reward curves for all experiments.

        Args:
            reward_df: DataFrame with reward curves.
            save_path: Optional path to save the plot.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for experiment in reward_df["experiment"].unique():
            exp_data = reward_df[reward_df["experiment"] == experiment]
            ax.plot(
                exp_data["timestep"],
                exp_data["average_reward"],
                label=experiment,
                linewidth=2,
            )

        ax.set_xlabel("Timestep", fontsize=12)
        ax.set_ylabel("Average Reward", fontsize=12)
        ax.set_title("Moving Average Reward Comparison", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_comparison_bars(
        self,
        comparison_df: pd.DataFrame,
        metric: str = "cumulative_regret",
        save_path: Optional[Path] = None,
    ) -> None:
        """Create bar plot comparing a specific metric across experiments.

        Args:
            comparison_df: DataFrame with comparison metrics.
            metric: Metric to plot.
            save_path: Optional path to save the plot.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        sorted_df = comparison_df.sort_values(metric)

        bars = ax.barh(sorted_df["experiment"], sorted_df[metric])

        for i, bar in enumerate(bars):
            color_intensity = i / len(bars)
            bar.set_color(plt.cm.viridis(color_intensity))

        ax.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel("Experiment", fontsize=12)
        ax.set_title(
            f"{metric.replace('_', ' ').title()} Comparison",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_interactive_regret(
        self,
        regret_df: pd.DataFrame,
        save_path: Optional[Path] = None,
    ) -> None:
        """Create interactive Plotly plot of regret curves.

        Args:
            regret_df: DataFrame with regret curves.
            save_path: Optional path to save the plot.
        """
        fig = px.line(
            regret_df,
            x="timestep",
            y="cumulative_regret",
            color="experiment",
            title="Interactive Cumulative Regret Comparison",
            labels={
                "timestep": "Timestep",
                "cumulative_regret": "Cumulative Regret",
                "experiment": "Experiment",
            },
        )

        fig.update_layout(
            hovermode="x unified",
            template="plotly_white",
            font={"size": 12},
            title_font_size=16,
        )

        if save_path:
            fig.write_html(str(save_path))

        fig.show()

    def create_dashboard(
        self,
        comparison_df: pd.DataFrame,
        regret_df: pd.DataFrame,
        reward_df: pd.DataFrame,
        save_path: Optional[Path] = None,
    ) -> None:
        """Create a comprehensive dashboard with multiple plots.

        Args:
            comparison_df: DataFrame with comparison metrics.
            regret_df: DataFrame with regret curves.
            reward_df: DataFrame with reward curves.
            save_path: Optional path to save the dashboard.
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Cumulative Regret",
                "Average Reward",
                "Total Reward Comparison",
                "Average Regret Comparison",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}],
            ],
        )

        for experiment in regret_df["experiment"].unique():
            exp_regret = regret_df[regret_df["experiment"] == experiment]
            fig.add_trace(
                go.Scatter(
                    x=exp_regret["timestep"],
                    y=exp_regret["cumulative_regret"],
                    name=experiment,
                    mode="lines",
                ),
                row=1,
                col=1,
            )

            exp_reward = reward_df[reward_df["experiment"] == experiment]
            fig.add_trace(
                go.Scatter(
                    x=exp_reward["timestep"],
                    y=exp_reward["average_reward"],
                    name=experiment,
                    mode="lines",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        sorted_total = comparison_df.sort_values("total_reward", ascending=True)
        fig.add_trace(
            go.Bar(
                y=sorted_total["experiment"],
                x=sorted_total["total_reward"],
                orientation="h",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        sorted_regret = comparison_df.sort_values("average_regret", ascending=True)
        fig.add_trace(
            go.Bar(
                y=sorted_regret["experiment"],
                x=sorted_regret["average_regret"],
                orientation="h",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            height=900,
            title_text="Contextual Bandits Performance Dashboard",
            title_font_size=20,
            template="plotly_white",
        )

        fig.update_xaxes(title_text="Timestep", row=1, col=1)
        fig.update_xaxes(title_text="Timestep", row=1, col=2)
        fig.update_xaxes(title_text="Total Reward", row=2, col=1)
        fig.update_xaxes(title_text="Average Regret", row=2, col=2)

        fig.update_yaxes(title_text="Cumulative Regret", row=1, col=1)
        fig.update_yaxes(title_text="Average Reward", row=1, col=2)

        if save_path:
            fig.write_html(str(save_path))

        fig.show()
