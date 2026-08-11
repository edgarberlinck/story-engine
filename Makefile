# Makefile for Story Engine Project

# Default target
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make test     - Run all tests"
	@echo "  make watch    - Watch for test changes"
	@echo "  make install  - Install dependencies and download models"
	@echo "  make models   - Download models only"
	@echo "  make models-ui - Interactive TUI to manage model downloads"
	@echo "  make benchmark-image - Run image generation benchmark suite"
	@echo "  make benchmark-video - Run image-to-video generation benchmark suite"
	@echo "  make benchmark-face  - Run face recognition benchmark (writes report)"
	@echo "  make benchmark-face-sdxl - Run face recognition benchmark with SDXL"
	@echo "  make benchmark_fullbody_recognition - Run full body face recognition benchmark"

# UI targets
.PHONY: ui cli
ui:
	@echo "Launching Story Engine UI..."
	.venv/bin/python ui/main.py

cli:
	@echo "Story Engine CLI"
	.venv/bin/python cli/main.py --help

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

# Interactive model download manager (Rust TUI)
.PHONY: models-ui
models-ui:
	@cargo build --release --manifest-path dev/models-ui/Cargo.toml
	@./dev/models-ui/target/release/models-ui .

# Run the image generation benchmark suite
.PHONY: benchmark-image
benchmark-image:
	@echo "Running image generation benchmark..."
	python generators/benchmark_image_generator.py

# Run the image-to-video generation benchmark suite
.PHONY: benchmark-video
benchmark-video:
	@echo "Running image-to-video generation benchmark..."
	python generators/benchmark_video_generator.py

# Run the face recognition benchmark suite
.PHONY: benchmark-face
benchmark-face:
	@echo "Running face recognition benchmark..."
	python generators/benchmark_face_recognition.py

# Run the face recognition benchmark suite with SDXL
.PHONY: benchmark-face-sdxl
benchmark-face-sdxl:
	@echo "Running face recognition benchmark (SDXL)..."
	python generators/benchmark_face_recognition_sdxl.py

# Run the full body face recognition benchmark suite
.PHONY: benchmark_fullbody_recognition
benchmark_fullbody_recognition:
	@echo "Running full body face recognition benchmark..."
	python generators/benchmark_fullbody_recognition.py

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