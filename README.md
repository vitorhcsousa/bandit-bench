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
python -m cb_comparison.cli
```

Or use the Makefile:

```bash
make run
```

## Project Structure

```
bandit-bench/
├── src/cb_comparison/          # Main source code
│   ├── bandits/                # Bandit implementations
│   │   ├── cb_library.py       # Contextual Bandits Library wrapper
│   │   ├── pytorch_bandit.py   # PyTorch bandits wrapper
│   │   ├── river_bandit.py     # River bandits wrapper
│   │   └── vowpal_bandit.py    # Vowpal Wabbit wrapper
│   ├── data/                   # Dataset management
│   ├── evaluation/             # Metrics and comparison tools
│   └── utils/                  # Visualization utilities
├── experiments/results/        # Experiment outputs
├── tests/                      # Unit tests
└── pyproject.toml             # Project configuration
```

## Usage

### Running Experiments

```python
from cb_comparison.evaluation.comparison import run_comparison
from cb_comparison.data.dataset import load_dataset

# Load dataset
dataset = load_dataset("my_dataset")

# Run comparison
results = run_comparison(
    dataset=dataset,
    algorithms=["vowpal", "pytorch", "river"],
    n_rounds=1000
)
```

### Visualizing Results

```python
from cb_comparison.utils.visualization import plot_regret_curves

# Generate plots
plot_regret_curves("experiments/results/regret_curves.csv")
```

## Development

Run tests:
```bash
make test
```

Run linting:
```bash
make lint
```

Format code:
```bash
make format
```

## Results

Experiment results are saved in `experiments/results/` including:
- CSV files with detailed metrics
- PNG plots comparing algorithm performance
- Interactive HTML dashboards

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

See `LICENSE` file for details.

## Requirements

- Python 3.10+
- Dependencies managed via `uv` and `pyproject.toml`

## Contact

For issues and questions, please open an issue on GitHub.