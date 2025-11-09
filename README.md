# Contextual Bandits Comparison

A comprehensive framework for comparing contextual bandit implementations and exploration algorithms across multiple libraries.

## Overview

This project provides a unified interface for evaluating and comparing different contextual bandit algorithms from various libraries including:

- **PyTorch**: Neural network-based contextual bandits with custom implementations
- **Vowpal Wabbit**: Industry-standard online learning platform
- **River**: Online machine learning library
- **Contextualbandits**: Specialized library for bandit algorithms

### Key Features

- 🎯 **Unified Interface**: Common API across all implementations
- 📊 **Multiple Exploration Strategies**: Epsilon-greedy, UCB, Thompson Sampling, LinUCB, and more
- 🔬 **Comprehensive Evaluation**: Built-in metrics including cumulative regret, average reward, and action distribution
- 📈 **Rich Visualizations**: Interactive and static plots for analysis
- 🛠️ **CLI Tool**: Easy-to-use command-line interface with Typer
- 🧪 **Synthetic Dataset**: Message-feedback simulation for realistic testing

## Installation

### Requirements

- **Python 3.9, 3.10, or 3.11** (for best Vowpal Wabbit compatibility)
- **Operating System**: Linux, macOS, or Windows
- **System Dependencies** (for Vowpal Wabbit):
  - Linux: `build-essential`, `cmake`, `libboost-all-dev`
  - macOS: Xcode Command Line Tools, `cmake`, `boost`
  - Windows: Visual Studio Build Tools

### macOS Quick Install 🍎

For macOS users, we provide an automated installation script:

```bash
# Download or extract the project
cd bandit-bench

# Run the automated installer
chmod +x install_macos.sh
./install_macos.sh
```

This script will:
- Install Homebrew (if needed)
- Install Python 3.11 (if needed)
- Install Xcode Command Line Tools (if needed)
- Install Boost libraries
- Install uv package manager
- Create virtual environment
- Install all dependencies including Vowpal Wabbit
- Run a test to verify installation

**For manual installation or troubleshooting, see the [macOS Setup Guide](MACOS_SETUP.md).**

### Step 1: Install System Dependencies (Manual Installation)

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libboost-all-dev zlib1g-dev
```

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake boost zlib
```

#### Windows
1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
   - Select "Desktop development with C++"
2. Install [CMake](https://cmake.org/download/)
3. Add CMake to PATH

For detailed Vowpal Wabbit installation instructions, see our [VOWPAL_WABBIT_INSTALLATION.md](VOWPAL_WABBIT_INSTALLATION.md) guide or the [official VW documentation](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Python).

### Step 2: Install Python Package Manager

#### Using uv (Recommended)

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Restart your shell after installation
```

#### Or use pip (if you prefer)
Python 3.10+ should come with pip installed.

### Step 3: Install Project

#### Using uv
```bash
# Clone repository
git clone <repository-url>
cd bandit-bench

# Create virtual environment with Python 3.10+
uv venv --python 3.10

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies
uv pip install -e ".[dev]"
```

#### Using pip
```bash
# Clone repository
git clone <repository-url>
cd bandit-bench

# Ensure Python 3.10+ is active
python --version

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### Step 4: Verify Installation

```bash
# Test Vowpal Wabbit installation
python -c "from vowpalwabbit import pyvw; print('✓ Vowpal Wabbit:', pyvw.__version__)"

# Test PyTorch installation
python -c "import torch; print('✓ PyTorch:', torch.__version__)"

# Test other dependencies
python -c "import river; print('✓ River: OK')"
python -c "import contextualbandits; print('✓ Contextualbandits: OK')"

# Run the CLI help
cb-compare --help
```

### Troubleshooting

#### Vowpal Wabbit Installation Issues

**Issue**: `ModuleNotFoundError: No module named 'vowpalwabbit'`
```bash
# Try installing from conda-forge (alternative method)
conda install -c conda-forge vowpalwabbit

# Or build from source
pip install vowpalwabbit --no-binary vowpalwabbit
```

**Issue**: Compilation errors on macOS
```bash
# Ensure Xcode Command Line Tools are installed
xcode-select --install

# Update boost
brew upgrade boost
```

**Issue**: Windows compilation errors
- Ensure Visual Studio Build Tools are properly installed
- Restart terminal after installing CMake
- Consider using Windows Subsystem for Linux (WSL)

For more help, see:
- [Vowpal Wabbit Python Wiki](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Python)
- [Vowpal Wabbit Issues](https://github.com/VowpalWabbit/vowpal_wabbit/issues)

## Quick Start

### Command Line Usage

Run a comparison with default settings:

```bash
cb-compare run
```

Run with specific libraries and exploration algorithms:

```bash
cb-compare run \
    --n-actions 5 \
    --n-rounds 2000 \
    --libraries pytorch vowpal \
    --explorations epsilon-greedy ucb thompson
```

List all available options:

```bash
cb-compare list-options
```

Generate visualizations from existing results:

```bash
cb-compare visualize --results-dir experiments/results
```

### Python API Usage

```python
from cb_comparison.data import MessageFeedbackDataset
from cb_comparison.bandits import PyTorchBandit, VowpalWabbitBandit
from cb_comparison.evaluation import BanditComparison
from cb_comparison.utils import BanditVisualizer

# Create dataset
dataset = MessageFeedbackDataset(
    n_actions=5,
    context_dim=10,
    random_seed=42
)

# Initialize comparison
comparison = BanditComparison(dataset)

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

# Run experiments
comparison.run_all()

# Get results
results_df = comparison.get_comparison_dataframe()
print(results_df)

# Visualize
visualizer = BanditVisualizer()
regret_df = comparison.get_regret_curves()
visualizer.plot_regret_curves(regret_df)
```

## Architecture

### Project Structure

```
bandit-bench/
├── src/cb_comparison/
│   ├── bandits/              # Bandit implementations
│   │   ├── base.py          # Abstract base class
│   │   ├── pytorch_bandit.py
│   │   ├── vowpal_bandit.py
│   │   ├── river_bandit.py
│   │   └── cb_library.py
│   ├── data/                 # Dataset generation
│   │   └── dataset.py
│   ├── evaluation/           # Metrics and comparison
│   │   ├── metrics.py
│   │   └── comparison.py
│   ├── utils/                # Utilities
│   │   └── visualization.py
│   └── cli.py               # CLI interface
├── tests/                    # Test suite
├── experiments/results/      # Experiment outputs
├── pyproject.toml
└── README.md
```

## Available Algorithms

### Libraries

| Library | Description | Strengths |
|---------|-------------|-----------|
| **PyTorch** | Neural network implementation | Flexible, GPU support, deep learning |
| **Vowpal Wabbit** | Industry-standard online learning | Scalable, efficient, production-ready |
| **River** | Online ML library | Streaming data, lightweight |
| **Contextualbandits** | Specialized bandit library | Research-oriented, comprehensive |

### Exploration Strategies

#### PyTorch & River
- **Epsilon-Greedy**: Balance exploration/exploitation with probability ε
- **UCB (Upper Confidence Bound)**: Optimistic action selection
- **Thompson Sampling**: Bayesian probability matching

#### Vowpal Wabbit
- **Epsilon-Greedy**: Classic exploration strategy
- **Bag**: Bootstrap aggregation for exploration
- **Cover**: Coverage-based exploration
- **RegCB**: Regression-based contextual bandits

#### Contextualbandits Library
- **LinUCB**: Linear UCB with feature uncertainty
- **Thompson Sampling**: Bootstrapped Thompson sampling
- **Epsilon-Greedy**: With base learner
- **Softmax**: Temperature-based exploration

## Evaluation Metrics

The framework tracks multiple performance metrics:

- **Cumulative Regret**: Total difference between optimal and actual rewards
- **Average Regret**: Mean regret per round
- **Total Reward**: Sum of all received rewards
- **Action Distribution**: Frequency of action selections
- **Optimal Action Rate**: Percentage of optimal action selections

## Visualization

The framework provides rich visualizations:

1. **Cumulative Regret Curves**: Compare learning efficiency
2. **Moving Average Reward**: Track reward trends
3. **Comparison Bar Charts**: Side-by-side metric comparison
4. **Interactive Dashboards**: Plotly-based comprehensive views

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Linting
ruff check src/

# Type checking
mypy src/

# Format check
ruff format --check src/
```

### Linting Configuration

The project uses Ruff for fast Python linting with the following checks:
- pycodestyle (E, W)
- pyflakes (F)
- isort (I)
- pep8-naming (N)
- pyupgrade (UP)
- flake8-bugbear (B)
- flake8-comprehensions (C4)
- flake8-simplify (SIM)

## Use Cases

This framework is ideal for:

- **Algorithm Research**: Compare novel exploration strategies
- **System Design**: Evaluate bandits for production systems
- **Education**: Learn contextual bandit algorithms
- **Benchmarking**: Test performance on custom datasets
- **Hyperparameter Tuning**: Find optimal configurations

## Example Scenarios

### Marketing Messages

```python
dataset = MessageFeedbackDataset(
    n_actions=5,  # 5 message variants
    context_dim=10  # User features
)

# Each action represents a different message variant
# Context includes user demographics, behavior, preferences
# Reward is user engagement (click, conversion, etc.)
```

### Content Recommendation

```python
# Actions: Different content categories or items
# Context: User profile, time, location, past behavior
# Reward: Engagement metric (watch time, likes, shares)
```

## Performance Considerations

- **PyTorch**: GPU acceleration for large-scale experiments
- **Vowpal Wabbit**: Memory-efficient for massive datasets
- **River**: Ideal for streaming/online scenarios
- **Contextualbandits**: Rich algorithm selection

## Contributing

Contributions welcome! Areas for improvement:

- Additional bandit implementations
- More exploration strategies
- Enhanced visualization options
- Performance optimizations
- Extended documentation

## References

- [Vowpal Wabbit Documentation](https://vowpalwabbit.org/)
- [PyTorch Documentation](https://pytorch.org/)
- [River Documentation](https://riverml.xyz/)
- [Contextual Bandits Paper](https://arxiv.org/abs/1003.0146)

## Acknowledgments

Built with modern Python tools:
- [uv](https://github.com/astral-sh/uv) for package management
- [Typer](https://typer.tiangolo.com/) for CLI
- [Rich](https://rich.readthedocs.io/) for terminal formatting
- [Plotly](https://plotly.com/) for interactive visualizations
