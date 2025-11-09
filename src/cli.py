from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from bandits import (
    ContextualBanditsWrapper,
    PyTorchBandit,
    RiverBandit,
)

try:
    from bandits import VowpalWabbitBandit

    VOWPAL_AVAILABLE = True
except ImportError:
    VOWPAL_AVAILABLE = False

from data import MessageFeedbackDataset
from evaluation import BanditComparison
from utils import BanditVisualizer

app = typer.Typer(
    name="cb-compare",
    help="Contextual Bandits Comparison Tool",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    n_actions: Annotated[
        int, typer.Option("--actions", "-a", help="Number of message variants/actions")
    ] = 5,
    context_dim: Annotated[
        int, typer.Option("--features", "-f", help="Dimension of context features")
    ] = 10,
    n_rounds: Annotated[
        int, typer.Option("--rounds", "-r", help="Number of rounds per experiment")
    ] = 1000,
    libraries: Annotated[
        Optional[list[str]],
        typer.Option(
            "--libraries",
            "-l",
            help="Libraries to compare (pytorch, vowpal, river, cb-library). Repeat flag for multiple.",
        ),
    ] = None,
    explorations: Annotated[
        Optional[list[str]],
        typer.Option(
            "--explorations",
            "-e",
            help="Exploration algorithms to test. Repeat flag for multiple.",
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory to save results"),
    ] = Path("experiments/results"),
    seed: Annotated[int, typer.Option("--seed", help="Random seed for reproducibility")] = 42,
    show_plots: Annotated[
        bool,
        typer.Option("--show-plots/--no-show-plots", help="Show plots after experiments"),
    ] = True,
) -> None:
    """Run contextual bandits comparison experiments.

    Args:
        n_actions: Number of message variants/actions to compare.
        context_dim: Dimensionality of context feature vectors.
        n_rounds: Number of rounds to run each experiment.
        libraries: list of libraries to compare.
        explorations: list of exploration algorithms to test.
        output_dir: Directory where results will be saved.
        seed: Random seed for reproducibility.
        show_plots: Whether to display plots after completion.
    """
    console.print("\n[bold cyan]Contextual Bandits Comparison[/bold cyan]\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[green]✓[/green] Creating dataset with {n_actions} actions")
    dataset = MessageFeedbackDataset(
        n_actions=n_actions,
        context_dim=context_dim,
        random_seed=seed,
    )

    comparison = BanditComparison(dataset, random_seed=seed)

    selected_libraries = libraries or ["pytorch", "vowpal", "river", "cb-library"]

    console.print(f"\n[green]✓[/green] Selected libraries: {', '.join(selected_libraries)}")

    if "pytorch" in selected_libraries:
        exploration_algos = explorations or ["epsilon-greedy", "ucb", "thompson"]
        for algo in exploration_algos:
            comparison.add_experiment(
                name=f"PyTorch-{algo}",
                bandit=PyTorchBandit(
                    n_actions=n_actions,
                    input_dim=context_dim,
                    exploration_algorithm=algo,
                ),
                n_rounds=n_rounds,
            )

    if "vowpal" in selected_libraries:
        if not VOWPAL_AVAILABLE:
            console.print(
                "[yellow]Warning: Vowpal Wabbit not available. Install with: uv sync --extra vowpal[/yellow]"
            )
        else:
            exploration_algos = explorations or ["epsilon-greedy", "bag", "cover"]
            for algo in exploration_algos:
                comparison.add_experiment(
                    name=f"Vowpal-{algo}",
                    bandit=VowpalWabbitBandit(
                        n_actions=n_actions,
                        exploration_algorithm=algo,
                    ),
                    n_rounds=n_rounds,
                )

    if "river" in selected_libraries:
        exploration_algos = explorations or ["epsilon-greedy", "ucb", "thompson"]
        for algo in exploration_algos:
            comparison.add_experiment(
                name=f"River-{algo}",
                bandit=RiverBandit(
                    n_actions=n_actions,
                    exploration_algorithm=algo,
                ),
                n_rounds=n_rounds,
            )

    if "cb-library" in selected_libraries:
        exploration_algos = explorations or ["linucb", "thompson", "epsilon-greedy"]
        for algo in exploration_algos:
            comparison.add_experiment(
                name=f"CBLib-{algo}",
                bandit=ContextualBanditsWrapper(
                    n_actions=n_actions,
                    exploration_algorithm=algo,
                ),
                n_rounds=n_rounds,
            )

    console.print(f"\n[yellow]Running {len(comparison.experiments)} experiments...[/yellow]\n")
    comparison.run_all(show_progress=True)

    comparison_df = comparison.get_comparison_dataframe()
    regret_df = comparison.get_regret_curves()
    reward_df = comparison.get_reward_curves()

    csv_path = output_dir / "comparison_results.csv"
    comparison_df.to_csv(csv_path, index=False)
    console.print(f"\n[green]✓[/green] Results saved to {csv_path}")

    regret_csv_path = output_dir / "regret_curves.csv"
    regret_df.to_csv(regret_csv_path, index=False)
    console.print(f"[green]✓[/green] Regret curves saved to {regret_csv_path}")

    _display_results_table(comparison_df)

    if show_plots:
        console.print("\n[yellow]Generating visualizations...[/yellow]\n")
        visualizer = BanditVisualizer()

        visualizer.plot_regret_curves(
            regret_df,
            save_path=output_dir / "regret_curves.png",
        )

        visualizer.plot_reward_curves(
            reward_df,
            save_path=output_dir / "reward_curves.png",
        )

        visualizer.plot_comparison_bars(
            comparison_df,
            metric="cumulative_regret",
            save_path=output_dir / "regret_comparison.png",
        )

        visualizer.create_dashboard(
            comparison_df,
            regret_df,
            reward_df,
            save_path=output_dir / "dashboard.html",
        )

        console.print(f"[green]✓[/green] Visualizations saved to {output_dir}")


@app.command()
def list_options() -> None:
    """list available libraries and exploration algorithms."""
    console.print("\n[bold cyan]Available Options[/bold cyan]\n")

    console.print("[bold yellow]Libraries:[/bold yellow]")
    libraries = [
        ("pytorch", "PyTorch neural network implementation"),
        ("vowpal", "Vowpal Wabbit online learning"),
        ("river", "River online machine learning"),
        ("cb-library", "Contextualbandits specialized library"),
    ]

    for lib, desc in libraries:
        console.print(f"  • {lib}: {desc}")

    console.print("\n[bold yellow]Exploration Algorithms:[/bold yellow]")
    console.print("\n  PyTorch, River:")
    for algo in ["epsilon-greedy", "ucb", "thompson"]:
        console.print(f"    • {algo}")

    console.print("\n  Vowpal Wabbit:")
    for algo in ["epsilon-greedy", "bag", "cover", "regcb"]:
        console.print(f"    • {algo}")

    console.print("\n  CB Library:")
    for algo in ["linucb", "thompson", "epsilon-greedy", "softmax"]:
        console.print(f"    • {algo}")

    console.print()


@app.command()
def visualize(
    results_dir: Annotated[
        Path,
        typer.Option("--results-dir", "-d", help="Directory containing results"),
    ] = Path("experiments/results"),
) -> None:
    """Generate visualizations from existing results.

    Args:
        results_dir: Directory containing experiment results CSV files.
    """
    console.print("\n[bold cyan]Generating Visualizations[/bold cyan]\n")

    comparison_path = results_dir / "comparison_results.csv"
    regret_path = results_dir / "regret_curves.csv"

    if not comparison_path.exists() or not regret_path.exists():
        console.print("[red]✗[/red] Results files not found. Run experiments first.")
        return

    import pandas as pd

    comparison_df = pd.read_csv(comparison_path)
    regret_df = pd.read_csv(regret_path)

    visualizer = BanditVisualizer()

    visualizer.plot_regret_curves(
        regret_df,
        save_path=results_dir / "regret_curves.png",
    )

    visualizer.plot_comparison_bars(
        comparison_df,
        metric="cumulative_regret",
        save_path=results_dir / "regret_comparison.png",
    )

    console.print(f"[green]✓[/green] Visualizations saved to {results_dir}")


def _display_results_table(comparison_df) -> None:
    """Display results in a formatted table.

    Args:
        comparison_df: DataFrame containing comparison results.
    """
    table = Table(title="\nExperiment Results", show_header=True, header_style="bold magenta")

    table.add_column("Experiment", style="cyan")
    table.add_column("Library", style="green")
    table.add_column("Exploration", style="yellow")
    table.add_column("Total Reward", justify="right")
    table.add_column("Cum. Regret", justify="right")
    table.add_column("Avg. Regret", justify="right")

    for _, row in comparison_df.iterrows():
        table.add_row(
            row["experiment"],
            row["library"],
            row["exploration"],
            f"{row['total_reward']:.2f}",
            f"{row['cumulative_regret']:.2f}",
            f"{row['average_regret']:.4f}",
        )

    console.print(table)


if __name__ == "__main__":
    app()
