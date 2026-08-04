"""Per-frame illumination histograms.

prior.compute derives each frame's target histogram from that same frame's
own non-masked pixels (see its module docstring for why) rather than a
trailing window of past clean frames, so this module owns only the raw
histogram computation.
"""
import numpy as np


def compute_illumination_histogram(luminance: np.ndarray, n_bins: int = 64) -> np.ndarray:
    histogram, _ = np.histogram(luminance, bins=n_bins, range=(0.0, 1.0))
    return histogram.astype(np.float64) / luminance.size
