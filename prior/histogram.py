"""Per-frame illumination histograms and time-axis smoothing.

Reuses Detector's own risk determination (RiskSegment / FlickerScore.uncertain)
instead of an independent flicker-frame classifier: `TargetHistogramSmoother`
just takes a boolean per frame ("is this frame risky or uncertain?") supplied
by the caller (see prior/compute.py) and excludes those frames' histograms
from the trailing-window average -- so the smoothed target represents what
illumination should look like absent the flicker itself, not a blend with it.
"""
import numpy as np


def compute_illumination_histogram(luminance: np.ndarray, n_bins: int = 64) -> np.ndarray:
    histogram, _ = np.histogram(luminance, bins=n_bins, range=(0.0, 1.0))
    return histogram.astype(np.float64) / luminance.size


class TargetHistogramSmoother:
    """Trailing-window average of a histogram sequence, excluding frames the
    caller marks risky/uncertain. Falls back to the most recently seen clean
    frame's histogram when the current window holds no clean frame; returns
    None until at least one clean frame has ever been seen."""

    def __init__(self, window_frames: int):
        self.window_frames = window_frames
        self._history: list[tuple[np.ndarray, bool]] = []
        self._last_clean: np.ndarray | None = None

    def update(self, histogram: np.ndarray, is_risky_or_uncertain: bool) -> np.ndarray | None:
        self._history.append((histogram, is_risky_or_uncertain))
        if len(self._history) > self.window_frames:
            self._history.pop(0)

        if not is_risky_or_uncertain:
            self._last_clean = histogram

        clean_in_window = [h for h, risky in self._history if not risky]
        if clean_in_window:
            return np.mean(clean_in_window, axis=0)
        return self._last_clean
