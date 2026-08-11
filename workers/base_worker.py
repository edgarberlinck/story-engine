"""
Base worker infrastructure for background tasks.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any, Dict
import threading


class GenerationWorker:
    """Simple thread pool based worker for heavy generation tasks."""

    def __init__(self, max_workers: int = 1):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures = {}
        self._lock = threading.Lock()

    def submit(self, task_fn: Callable, *args, **kwargs) -> Any:
        """Submit a task to background worker."""
        future = self.executor.submit(task_fn, *args, **kwargs)
        with self._lock:
            self._futures[id(future)] = future
        return future

    def shutdown(self):
        self.executor.shutdown(wait=True)


# Singleton instance for generation tasks
generation_worker = GenerationWorker(max_workers=1)
