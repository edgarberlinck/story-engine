import time
import psutil
import json
import os
from pathlib import Path

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

class ModelMetrics:
    def __init__(self):
        self.metrics = {}
        
    def start_timer(self):
        """Start timing for generation."""
        self.start_time = time.time()
        self.start_memory = get_memory_usage()
        
    def end_timer(self):
        """End timing and calculate duration and memory usage."""
        self.end_time = time.time()
        self.end_memory = get_memory_usage()
        self.duration_ms = int((self.end_time - self.start_time) * 1000)
        self.peak_memory_mb = int(self.end_memory - self.start_memory)
        
    def record_generation(self, model_name, prompt, seed, steps, cfg, width, height, output_path):
        """Record all generation metrics."""
        self.metrics = {
            "model": model_name,
            "prompt": prompt,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "width": width,
            "height": height,
            "duration_ms": self.duration_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "output": output_path
        }
        
    def save_metrics(self, filename_base, model_name):
        """Save metrics to a JSON file."""
        metrics_filename = f"outputs/{filename_base}_{model_name}_benchmark_metrics.json"
        with open(metrics_filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        return metrics_filename