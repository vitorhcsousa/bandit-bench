# Bandit-Bench

A comprehensive benchmarking framework for comparing contextual bandit algorithms across multiple libraries.

## Overview

Bandit-Bench provides a unified interface for evaluating and comparing contextual bandit implementations from popular Python libraries including:
- **Vowpal Wabbit** - Fast, production-ready contextual bandits
- **PyTorch Bandits** - Deep learning-based bandit algorithms
- **River** - Online machine learning bandits
- **Contextual Bandits Library** - Traditional bandit implementations

## Features

- 🎯 **Unified API** - Common interface across different bandit libraries
- 📊 **Comprehensive Metrics** - Regret, cumulative reward, and performance tracking
- 📈 **Visualization** - Interactive dashboards and comparison plots
- 🔄 **Simulation Engine** - Robust framework for running experiments
- 🧪 **Extensible** - Easy to add new algorithms and datasets
- 📦 **Dataset Management** - Built-in dataset handling and generation
- 💻 **CLI & Python API** - Use as a command-line tool or Python library

## Supported Libraries & Algorithms

### Libraries
- **PyTorch Bandits**: Neural network-based contextual bandits with deep learning
- **Vowpal Wabbit**: Fast, scalable online learning implementation
- **River**: Online machine learning with incremental algorithms
- **Contextual Bandits Library**: Specialized library with traditional implementations

### Exploration Algorithms

**PyTorch & River:**
- Epsilon-Greedy
- Upper Confidence Bound (UCB)
- Thompson Sampling

**Vowpal Wabbit:**
- Epsilon-Greedy
- Bagging (bag)
- Cover
- Regression CB (regcb)

**Contextual Bandits Library:**
- LinUCB
- Thompson Sampling
- Epsilon-Greedy
- Softmax

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd bandit-bench
```

2. Install dependencies using `uv` (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

## Quick Start

Run a comparison experiment:

```bash
bandit-bench run
```

Or use the Makefile:

```bash
make run
```

To see all available options:

```bash
bandit-bench list-options
```

## Project Structure

```
bandit-bench/
├── src/
│   ├── bandits/                # Bandit implementations
│   │   ├── base.py             # Abstract base class for contextual bandits
│   │   ├── cb_library.py       # Contextual Bandits Library wrapper
│   │   ├── pytorch_bandit.py   # PyTorch bandits wrapper
│   │   ├── river_bandit.py     # River bandits wrapper
│   │   └── vowpal_bandit.py    # Vowpal Wabbit wrapper
│   ├── data/                   # Dataset management
│   │   └── dataset.py          # MessageFeedbackDataset implementation
│   ├── evaluation/             # Metrics and comparison tools
│   │   ├── comparison.py       # BanditComparison framework
│   │   ├── metrics.py          # Performance metrics
│   │   └── visualizer.py       # Result visualization
│   ├── utils/                  # Utility functions
│   │   └── visualization.py    # BanditVisualizer for plotting
│   ├── core/                   # Core abstractions
│   │   └── base.py             # Base models and interfaces
│   ├── algorithms/             # Alternative algorithm implementations
│   └── cli.py                  # Command-line interface
├── experiments/results/        # Experiment outputs
├── tests/                      # Unit tests
├── Makefile                    # Build and task automation
└── pyproject.toml             # Project configuration
```

## Usage

### Running Experiments

Using the CLI:

```bash
# Basic usage
bandit-bench run --rounds 1000 --features 10 --actions 5

# Compare specific libraries
bandit-bench run --libraries pytorch --libraries vowpal --rounds 5000

# Use specific exploration algorithms
bandit-bench run --explorations epsilon-greedy --explorations ucb
```

Using Python API:

```python
from data import MessageFeedbackDataset
from evaluation import BanditComparison
from bandits import PyTorchBandit, VowpalWabbitBandit, RiverBandit

# Create dataset
dataset = MessageFeedbackDataset(
    n_actions=5,
    context_dim=10,
    random_seed=42
)

# Initialize comparison framework
comparison = BanditComparison(dataset, random_seed=42)

# Add experiments
comparison.add_experiment(
    name="PyTorch-UCB",
    bandit=PyTorchBandit(
        n_actions=5,
        input_dim=10,
        exploration_algorithm="ucb"
    ),
    n_rounds=1000
)

comparison.add_experiment(
    name="Vowpal-Epsilon",
    bandit=VowpalWabbitBandit(
        n_actions=5,
        exploration_algorithm="epsilon-greedy"
    ),
    n_rounds=1000
)

# Run all experiments
comparison.run_all(show_progress=True)

# Get results
results_df = comparison.get_comparison_dataframe()
regret_df = comparison.get_regret_curves()
```

### Visualizing Results

Using the CLI:

```bash
bandit-bench visualize --results-dir experiments/results
```

Using Python API:

```python
import pandas as pd
from utils import BanditVisualizer

# Load results
regret_df = pd.read_csv("experiments/results/regret_curves.csv")
comparison_df = pd.read_csv("experiments/results/comparison_results.csv")
reward_df = pd.read_csv("experiments/results/reward_curves.csv")

# Create visualizer
visualizer = BanditVisualizer()

# Generate plots
visualizer.plot_regret_curves(regret_df, save_path="regret.png")
visualizer.plot_reward_curves(reward_df, save_path="reward.png")
visualizer.plot_comparison_bars(comparison_df, metric="cumulative_regret", save_path="comparison.png")

# Create interactive dashboard
visualizer.create_dashboard(
    comparison_df,
    regret_df,
    reward_df,
    save_path="dashboard.html"
)
```

## CLI Commands

The `bandit-bench` CLI provides several commands:

### Run Experiments
```bash
bandit-bench run [OPTIONS]
```
Options:
- `--actions, -a`: Number of message variants/actions (default: 5)
- `--features, -f`: Dimension of context features (default: 10)
- `--rounds, -r`: Number of rounds per experiment (default: 1000)
- `--libraries, -l`: Libraries to compare (pytorch, vowpal, river, cb-library)
- `--explorations, -e`: Exploration algorithms to test
- `--output-dir, -o`: Directory to save results (default: experiments/results)
- `--seed`: Random seed for reproducibility (default: 42)
- `--show-plots/--no-show-plots`: Show plots after experiments (default: True)

### List Available Options
```bash
bandit-bench list-options
```
Shows all available libraries and exploration algorithms.

### Visualize Results
```bash
bandit-bench visualize [OPTIONS]
```
Options:
- `--results-dir, -d`: Directory containing results (default: experiments/results)

## Development

Run tests:
```bash
make test
```

Run tests with coverage:
```bash
make test-cov
```

Run linting:
```bash
make lint
```

Fix linting issues:
```bash
make lint-fix
```

Format code:
```bash
make format
```

Type checking:
```bash
make type-check
```

Run all quality checks:
```bash
make qa
```

## Results

Experiment results are saved in `experiments/results/` including:
- **comparison_results.csv**: Summary metrics for all experiments
- **regret_curves.csv**: Cumulative regret over time for each experiment
- **reward_curves.csv**: Moving average rewards over time
- **regret_curves.png**: Visualization of cumulative regret
- **reward_curves.png**: Visualization of reward trends
- **regret_comparison.png**: Bar chart comparing final regret across algorithms
- **dashboard.html**: Interactive Plotly dashboard with all metrics

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

See `LICENSE` file for details.

## Requirements

- Python 3.9 - 3.12
- Dependencies managed via `uv` and `pyproject.toml`
- Main dependencies:
  - PyTorch >= 2.0.0
  - Vowpal Wabbit >= 9.9.0
  - River >= 0.21.0
  - contextualbandits >= 0.3.0
  - numpy, pandas, scikit-learn
  - matplotlib, seaborn, plotly (for visualization)
  - typer, rich (for CLI)

## Contact

For issues and questions, please open an issue on GitHub.