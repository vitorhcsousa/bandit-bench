# Minimal settings
PKG      := cb_comparison
SRC      := src
TESTS    := tests
CLI      := cb-compare
PY       := python3

ROUNDS   ?= 5000
FEATURES ?= 10
ACTIONS  ?= 5

.PHONY: install format check lint lint-fix type-check test test-cov test-fast run clean clean-build build lock setup-vw check-vw qa info

install:
	uv pip install -e ".[dev]"

# --- keep this format flow ---
format:
	@echo "▶ Running ruff format (code formatting)"
	ruff format $(RUFF_ARGS) $(PKG) $(TESTS)
	@echo "▶ Organizing imports with ruff (isort rules)"
	ruff check --fix --select I $(RUFF_ARGS) $(PKG) $(TESTS)

check:
	@echo "▶ Checking formatting"
	ruff format --check $(RUFF_ARGS) $(PKG) $(TESTS)
	@echo "▶ Linting with ruff"
	ruff check $(RUFF_ARGS) $(PKG) $(TESTS)
	@echo "▶ Type checking with mypy"
	mypy $(MYPY_ARGS) $(PKG)
	@echo "▶ Running tests with pytest" pytest $(PYTEST_ARGS)

lint:
	uv run ruff check $(SRC) $(TESTS)

lint-fix:
	uv run ruff check --fix $(SRC) $(TESTS)

type-check:
	uv run mypy $(SRC)

test:
	PYTHONPATH=$(SRC) uv run pytest

test-cov:
	PYTHONPATH=$(SRC) uv run pytest --cov=$(PKG) --cov-report=html --cov-report=term-missing

test-fast:
	PYTHONPATH=$(SRC) uv run pytest -m "not slow and not integration"

run:
	uv run $(CLI) run --rounds $(ROUNDS) --features $(FEATURES) --actions $(ACTIONS) $(ARGS)

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
