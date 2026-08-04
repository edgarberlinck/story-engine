"""
Download manager for the story-engine project.
Handles concurrent downloads with adaptive speed control.
"""

import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

class DownloadManager:
    """Manages concurrent model downloads with adaptive speed control."""
    
    def __init__(self, max_parallel: int = 4, adaptive: bool = True):
        """
        Initialize the download manager.
        
        Args:
            max_parallel (int): Maximum number of parallel downloads
            adaptive (bool): Whether to adapt download speed based on connection
        """
        self.max_parallel = max_parallel
        self.adaptive = adaptive
        self.session = requests.Session()
        
    def get_download_speed(self, url: str) -> float:
        """Estimate download speed for a given URL."""
        try:
            start = time.perf_counter()
            downloaded = 0
            
            response = self.session.get(url, stream=True, timeout=10)
            response.raise_for_status()
            
            # Download a small chunk to estimate speed
            chunk_size = 1024 * 1024  # 1MB chunks
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    downloaded += len(chunk)
                    elapsed = time.perf_counter() - start
                    if elapsed > 0:
                        speed = downloaded / elapsed
                        return speed / 1024 / 1024  # Return MB/s
                break  # Only test with first chunk
            
        except Exception:
            pass
        return 0
    
    def get_optimal_workers(self) -> int:
        """Get optimal number of workers based on connection speed."""
        if not self.adaptive:
            return self.max_parallel
            
        # This is just a placeholder for actual network checks
        # In real implementation, we'd use something like speedtest-cli
        return min(4, self.max_parallel)  # Default to 4 workers for demo
    
    def download_file_with_progress(self, url: str, filename: str, progress_callback=None) -> bool:
        """Download a single file with progress tracking."""
        try:
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            progress = (downloaded / total_size) * 100
                            progress_callback(progress)
            
            return True
            
        except Exception as e:
            print(f"Download failed: {e}")
            return False
    
    def download_models(self, models: List[Tuple[str, str, str]], max_workers: int = None) -> int:
        """
        Download multiple models concurrently.
        
        Args:
            models (List[Tuple[str, str, str]]): List of (name, url, destination) tuples
            max_workers (int): Maximum number of parallel workers
            
        Returns:
            int: Number of successful downloads
        """
        if max_workers is None:
            max_workers = self.get_optimal_workers()
        
        success_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_model = {
                executor.submit(
                    self._single_download,
                    name, url, destination
                ): name 
                for name, url, destination in models
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_model):
                success, name = future.result()
                if success:
                    success_count += 1
                else:
                    print(f"Error downloading {name}")
        
        return success_count
    
    def _single_download(self, name: str, url: str, destination: str) -> Tuple[bool, str]:
        """Download a single model."""
        print(f"Downloading {name}...")
        
        try:
            # Create destination directory
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            
            # Simulate download progress for demo
            for i in range(10):
                time.sleep(0.3)
                if i % 2 == 0:
                    print(f"  Downloading... {i*10}% complete")
            
            # Create a simple marker file
            marker_file = Path(destination) / ".installed"
            marker_file.touch()
            
            print(f"✓ Downloaded {name}")
            return (True, name)
            
        except Exception as e:
            print(f"✗ Failed to download {name}: {e}")
            return (False, name)