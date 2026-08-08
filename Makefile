# Makefile for Story Engine Project

# Default target
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make test     - Run all tests"
	@echo "  make watch    - Watch for test changes"
	@echo "  make install  - Install dependencies and download models"
	@echo "  make models   - Download models only"
	@echo "  make generate - Generate an image"

# Test target
.PHONY: test
test:
	@echo "Running tests..."
	python -m unittest discover -s . -p "test_*.py" -v

# Watch test target
.PHONY: watch
watch:
	@echo "Watching for test changes..."
	watchmedo shell-command --recursive --command='python -m unittest discover -s . -p "test_*.py" -v' .

# Install dependencies and download models
.PHONY: install
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Downloading models..."
	python scripts/install.py

# Download models only
.PHONY: models
models:
	@echo "Downloading models..."
	python scripts/install.py

# Generate an image
.PHONY: generate
generate:
	@echo "Generating image..."
	python generators/image_generator.py

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