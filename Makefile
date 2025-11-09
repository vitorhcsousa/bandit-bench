# Minimal settings
PKG      := src
SRC      := src
TESTS    := tests
CLI      := cli
PY       := python3

ROUNDS   ?= 5000
FEATURES ?= 10
ACTIONS  ?= 5

.PHONY: install format check lint lint-fix type-check test test-cov test-fast run clean clean-build build lock setup-vw check-vw qa info

install:
	uv venv --python 3.10
	source .venv/bin/activate
	uv pip install -e ".[dev]"

# --- keep this format flow ---
format:
	@echo "▶ Running ruff format (code formatting)"
	ruff format $(RUFF_ARGS) $(SRC) $(TESTS)
	@echo "▶ Organizing imports with ruff (isort rules)"
	ruff check --fix --select I $(RUFF_ARGS) $(SRC) $(TESTS)

check:
	@echo "▶ Checking formatting"
	ruff format --check $(RUFF_ARGS) $(SRC) $(TESTS)
	@echo "▶ Linting with ruff"
	ruff check $(RUFF_ARGS) $(SRC) $(TESTS)
	@echo "▶ Type checking with mypy"

lint:
	uv run ruff check $(SRC) $(TESTS)

lint-fix:
	uv run ruff check --fix $(SRC) $(TESTS)

type-check:
	uv run mypy $(SRC)

test:
	PYTHONPATH=$(SRC) uv run pytest

test-cov:
	PYTHONPATH=$(SRC) uv run pytest --cov=src --cov-report=html --cov-report=term-missing

test-fast:
	PYTHONPATH=$(SRC) uv run pytest -m "not slow and not integration"

run:
	uv run bandit-bench run --rounds $(ROUNDS) --features $(FEATURES) --actions $(ACTIONS) $(ARGS)

clean:
	rm -rf .venv build/ dist/ *.egg-info .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ results/
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-build:
	rm -rf build/ dist/

build: clean-build
	uv pip install build >/dev/null
	$(PY) -m build --wheel

lock:
	mkdir -p dist
	uv pip freeze --exclude-editable > dist/constraints.txt

setup-vw:
	uv pip install "setuptools<81" "vowpalwabbit>=9.10.0,<9.11.0"


qa: lint lint-fix type-check test-cov

info:
	@echo "Package: $(PKG)"
	@echo "Source:  $(SRC)"
	@echo "Tests:   $(TESTS)"
	@echo "CLI:     $(CLI)"