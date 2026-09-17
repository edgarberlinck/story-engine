"""Tests for waveform peak downsampling (core/audio_peaks.py)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from core import audio_peaks


class TestComputePeaks(unittest.TestCase):
    def setUp(self):
        audio_peaks.clear_peak_cache()
        self.workdir = tempfile.TemporaryDirectory()
        self.wav = Path(self.workdir.name) / "tone.wav"
        sr = 24000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        sf.write(
            str(self.wav),
            (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
            sr,
        )

    def tearDown(self):
        self.workdir.cleanup()
        audio_peaks.clear_peak_cache()

    def test_returns_requested_bucket_count(self):
        peaks = audio_peaks.compute_peaks(self.wav, 50)
        self.assertEqual(len(peaks), 50)

    def test_peaks_bounded_and_symmetric_for_sine(self):
        peaks = audio_peaks.compute_peaks(self.wav, 20)
        for lo, hi in peaks:
            self.assertLessEqual(lo, hi)
            self.assertGreaterEqual(lo, -1.0)
            self.assertLessEqual(hi, 1.0)
        # A full-cycle sine bucket peaks near ±0.5.
        self.assertAlmostEqual(max(h for _, h in peaks), 0.5, places=1)
        self.assertAlmostEqual(min(low for low, _ in peaks), -0.5, places=1)

    def test_missing_file_and_bad_buckets(self):
        self.assertEqual(audio_peaks.compute_peaks("/nope.wav", 10), [])
        self.assertEqual(audio_peaks.compute_peaks(self.wav, 0), [])

    def test_cache_hit_returns_same_object(self):
        a = audio_peaks.compute_peaks(self.wav, 10)
        b = audio_peaks.compute_peaks(self.wav, 10)
        self.assertIs(a, b)


if __name__ == "__main__":
    unittest.main()
