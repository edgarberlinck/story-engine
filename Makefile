.PHONY: install models generate clean help

# Default target
all: help

# Install dependencies and download models
install: 
	@echo "Installing dependencies..."
	source .venv/bin/activate && pip install huggingface_hub torch diffusers pillow tqdm
	@echo "Installing models..."
	source .venv/bin/activate && python scripts/install.py

# Download models only
models:
	@echo "Downloading models..."
	source .venv/bin/activate && python scripts/install.py

# Generate an image
generate:
	@echo "Generating image for benchmarking..."
	source .venv/bin/activate && python generators/image_generator.py

# Clean up
clean:
	rm -rf models/diffusion/* models/segmentation/* models/text_generation/*
	@echo "Cleaned model directories"

# Show help
help:
	@echo "Available commands:"
	@echo "  install    - Install dependencies and download models"
	@echo "  models     - Download models only"
	@echo "  generate   - Generate an image"
	@echo "  clean      - Clean model directories"
	@echo "  help       - Show this help"
