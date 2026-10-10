.PHONY: install install-dev lint format format-check test typecheck coverage precommit clean

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

test:
	$(PYTHON) -m pytest -q

test-cov:
	$(PYTHON) -m pytest --cov=common --cov=web_app --cov-report=term-missing

typecheck:
	mypy

coverage-html:
	$(PYTHON) -m pytest --cov=common --cov=web_app --cov-report=html
	$(PYTHON) -m webbrowser htmlcov/index.html

precommit:
	pre-commit install

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage build dist *.egg-info