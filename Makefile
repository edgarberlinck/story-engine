# Makefile for Story Engine Project

# Default target
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make test     - Run all tests"
	@echo "  make test:watch  - Run tests in watch mode"

# Test target
.PHONY: test
test:
	@echo "Running tests..."
	python -m unittest discover -s tests -p "test_*.py" -v

# Watch test target
.PHONY: test:watch
test:watch:
	@echo "Watching for test changes..."
	watchmedo shell-command --recursive --command='python -m unittest discover -s tests -p "test_*.py" -v' tests/

# Install dependencies
.PHONY: install
install:
	pip install -r requirements.txt

# Clean build artifacts
.PHONY: clean
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -f story_engine.db

# Format code
.PHONY: format
format:
	black .

# Lint code
.PHONY: lint
lint:
	flake8 .

# All checks
.PHONY: check
check: lint test