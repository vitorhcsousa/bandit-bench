from pathlib import Path
from typing import Annotated, Literal, Optional

import typer
from rich.console import Console

from cb_comparison.algorithms.contextual_bandits_lib import ContextualBanditsLibBandit
from cb_comparison.algorithms.pytorch_bandits import PyTorchLinearBandit
from cb_comparison.algorithms.river_bandits import RiverBandit
from cb_comparison.algorithms.vowpal_bandits import VowpalWabbitBandit
from cb_comparison.core.base import BaseContextualBandit
from cb_comparison.data.dataset import DatasetConfig, MessageDataset
from cb_comparison.evaluation.simulator import BanditSimulator
from cb_comparison.evaluation.visualizer import ResultsVisualizer

app = typer.Typer(
    name="cb-compare",
    help="Contextual Bandits Comparison Tool",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    n_rounds: Annotated[
        int, typer.Option("--rounds", "-r", help="Number of simulation rounds")
    ] = 5000,
    n_features: Annotated[
        int, typer.Option("--features", "-f", help="Number of context features")
    ] = 10,
    n_actions: Annotated[int, typer.Option("--actions", "-a", help="Number of actions")] = 5,
    output_dir: Annotated[
        Path, typer.Option("--output", "-o", help="Output directory for results")
    ] = Path("results"),
    seed: Annotated[int, typer.Option("--seed", "-s", help="Random seed")] = 42,
    algorithms: Annotated[
        Optional[list[str]],
        typer.Option(
            "--algo",
            help="Algorithms to compare (repeatable): pytorch, vowpal, contextual, river, all",
            show_default="all",
        ),
    ] = None,
) -> None:
    console.print("[bold cyan]Contextual Bandits Comparison[/bold cyan]")
    console.print(f"Rounds: {n_rounds}, Features: {n_features}, Actions: {n_actions}\n")

    dataset_config = DatasetConfig(
        n_features=n_features,
        n_actions=n_actions,
        n_samples=n_rounds,
        random_seed=seed,
    )
    dataset = MessageDataset(dataset_config)

    bandits: list[BaseContextualBandit] = []

    # Normalizar os algoritmos seleccionados
    selected_algos = set(algorithms or ["all"])
    run_all = "all" in selected_algos

    if run_all or "pytorch" in selected_algos:
        bandits.extend(
            [
                PyTorchLinearBandit(n_actions, n_features, exploration="epsilon-greedy"),
                PyTorchLinearBandit(n_actions, n_features, exploration="ucb"),
                PyTorchLinearBandit(n_actions, n_features, exploration="thompson"),
                PyTorchLinearBandit(n_actions, n_features, exploration="softmax"),
            ]
        )

    if run_all or "vowpal" in selected_algos:
        bandits.extend(
            [
                VowpalWabbitBandit(n_actions, n_features, exploration="epsilon-greedy"),
                VowpalWabbitBandit(n_actions, n_features, exploration="bag"),
                VowpalWabbitBandit(n_actions, n_features, exploration="softmax"),
            ]
        )

    if run_all or "contextual" in selected_algos:
        bandits.extend(
            [
                ContextualBanditsLibBandit(n_actions, n_features, exploration="epsilon-greedy"),
                ContextualBanditsLibBandit(n_actions, n_features, exploration="softmax"),
                ContextualBanditsLibBandit(n_actions, n_features, exploration="bootstrap-ucb"),
                ContextualBanditsLibBandit(n_actions, n_features, exploration="bootstrap-ts"),
            ]
        )

    if run_all or "river" in selected_algos:
        bandits.extend(
            [
                RiverBandit(n_actions, n_features, exploration="epsilon-greedy"),
                RiverBandit(n_actions, n_features, exploration="ucb"),
                RiverBandit(n_actions, n_features, exploration="softmax"),
            ]
        )

    console.print(f"[green]Running simulation with {len(bandits)} algorithms...[/green]\n")

    simulator = BanditSimulator(dataset, seed=seed)
    simulator.run_simulation(bandits, n_rounds, verbose=True)

    console.print("\n[bold green]Simulation complete![/bold green]\n")
    simulator.print_results()

    visualizer = ResultsVisualizer(simulator.get_metrics())
    visualizer.create_all_plots(output_dir)
    visualizer.export_results_to_csv(output_dir / "results.csv")

    console.print(f"\n[bold blue]Results saved to {output_dir}[/bold blue]")


@app.command()
def compare_exploration(
    algorithm: Annotated[
        Literal["pytorch", "vowpal", "contextual", "river"],
        typer.Argument(help="Base algorithm: pytorch, vowpal, contextual, river"),
    ],
    n_rounds: Annotated[
        int, typer.Option("--rounds", "-r", help="Number of simulation rounds")
    ] = 5000,
    n_features: Annotated[
        int, typer.Option("--features", "-f", help="Number of context features")
    ] = 10,
    n_actions: Annotated[int, typer.Option("--actions", "-a", help="Number of actions")] = 5,
    output_dir: Annotated[
        Path, typer.Option("--output", "-o", help="Output directory for results")
    ] = Path("results"),
    seed: Annotated[int, typer.Option("--seed", "-s", help="Random seed")] = 42,
) -> None:
    console.print(f"[bold cyan]Comparing Exploration Strategies for {algorithm}[/bold cyan]\n")

    dataset_config = DatasetConfig(
        n_features=n_features,
        n_actions=n_actions,
        n_samples=n_rounds,
        random_seed=seed,
    )
    dataset = MessageDataset(dataset_config)

    bandits: list[BaseContextualBandit] = []

    if algorithm == "pytorch":
        bandits.extend(
            [
                PyTorchLinearBandit(n_actions, n_features, exploration="epsilon-greedy"),
                PyTorchLinearBandit(n_actions, n_features, exploration="ucb"),
                PyTorchLinearBandit(n_actions, n_features, exploration="thompson"),
                PyTorchLinearBandit(n_actions, n_features, exploration="softmax"),
            ]
        )
    elif algorithm == "vowpal":
        bandits.extend(
            [
                VowpalWabbitBandit(n_actions, n_features, exploration="epsilon-greedy"),
                VowpalWabbitBandit(n_actions, n_features, exploration="bag"),
                VowpalWabbitBandit(n_actions, n_features, exploration="softmax"),
            ]
        )
    elif algorithm == "contextual":
        bandits.extend(
            [
                ContextualBanditsLibBandit(n_actions, n_features, exploration="epsilon-greedy"),
                ContextualBanditsLibBandit(n_actions, n_features, exploration="softmax"),
                ContextualBanditsLibBandit(n_actions, n_features, exploration="bootstrap-ucb"),
                ContextualBanditsLibBandit(n_actions, n_features, exploration="bootstrap-ts"),
            ]
        )
    elif algorithm == "river":
        bandits.extend(
            [
                RiverBandit(n_actions, n_features, exploration="epsilon-greedy"),
                RiverBandit(n_actions, n_features, exploration="ucb"),
                RiverBandit(n_actions, n_features, exploration="softmax"),
            ]
        )
    else:
        console.print(f"[red]Unknown algorithm: {algorithm}[/red]")
        raise typer.Exit(1)

    simulator = BanditSimulator(dataset, seed=seed)
    simulator.run_simulation(bandits, n_rounds, verbose=True)

    console.print("\n[bold green]Simulation complete![/bold green]\n")
    simulator.print_results()

    output_algo_dir = output_dir / algorithm
    visualizer = ResultsVisualizer(simulator.get_metrics())
    visualizer.create_all_plots(output_algo_dir)
    visualizer.export_results_to_csv(output_algo_dir / "results.csv")

    console.print(f"\n[bold blue]Results saved to {output_algo_dir}[/bold blue]")


if __name__ == "__main__":
    app()
