"""Waveform peak extraction for timeline clip rendering.

Downsamples a WAV to per-pixel (min, max) peak pairs so the UI can draw an
Audacity-style waveform with QPainter. Results are cached per
(path, mtime, buckets) so scrolling/zooming never re-reads unchanged files.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

_PEAK_CACHE: Dict[Tuple[str, float, int], List[Tuple[float, float]]] = {}


def clear_peak_cache() -> None:
    _PEAK_CACHE.clear()


def compute_peaks(wav_path, num_buckets: int) -> List[Tuple[float, float]]:
    """Per-bucket (min, max) sample peaks in [-1, 1] for a WAV file.

    Returns an empty list when the file is missing/unreadable or
    ``num_buckets`` is not positive.
    """
    path = Path(wav_path)
    if num_buckets <= 0 or not path.exists():
        return []
    try:
        mtime = path.stat().st_mtime
        key = (str(path), mtime, num_buckets)
        cached = _PEAK_CACHE.get(key)
        if cached is not None:
            return cached

        import numpy as np
        import soundfile as sf

        data, _sr = sf.read(str(path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        n = len(mono)
        if n == 0:
            _PEAK_CACHE[key] = []
            return []
        buckets = min(num_buckets, n)
        edges = np.linspace(0, n, buckets + 1, dtype=int)
        peaks = [
            (float(chunk.min()), float(chunk.max()))
            for chunk in (mono[edges[i] : edges[i + 1]] for i in range(buckets))
            if len(chunk)
        ]
        _PEAK_CACHE[key] = peaks
        return peaks
    except Exception:  # noqa: BLE001
        logger.warning("Peak extraction failed for %s", path, exc_info=True)
        return []
