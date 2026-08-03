"""Wires Detector's extended run_detection_with_masks together with
prior.histogram: for every frame, produces a (smoothed target illumination
histogram, per-pixel risk mask) pair for Mitigator conditioning.

"Risky" for smoothing purposes uses Detector's *margin-padded* segments
(default margin_seconds=0.5, i.e. run_detection_with_masks's default) rather
than the zero-margin check DatasetSynth's validation uses -- Mitigator will
only ever receive margin-padded segments from BufferManager in production, so
PriorCalc's notion of "risky" should match what Mitigator actually sees.
"""
from dataclasses import dataclass

import numpy as np

from detector.luminance import relative_luminance
from detector.pipeline import run_detection_with_masks
from detector.profiles import ThresholdProfile
from prior.histogram import TargetHistogramSmoother, compute_illumination_histogram


@dataclass
class FramePrior:
    frame_index: int
    target_histogram: np.ndarray | None
    mask: np.ndarray


def compute_prior(
    frames: list[np.ndarray],
    fps: float,
    profile: ThresholdProfile,
    n_bins: int = 64,
) -> list[FramePrior]:
    scores, segments, masks = run_detection_with_masks(frames, fps=fps, profile=profile)

    risky_frame_indices = set()
    for segment in segments:
        risky_frame_indices.update(range(segment.start_frame, segment.end_frame + 1))

    window_frames = max(1, round(fps))
    smoother = TargetHistogramSmoother(window_frames=window_frames)

    results: list[FramePrior] = []
    for frame, score, mask in zip(frames, scores, masks):
        histogram = compute_illumination_histogram(relative_luminance(frame), n_bins=n_bins)
        is_risky_or_uncertain = score.frame_index in risky_frame_indices or score.uncertain
        target = smoother.update(histogram, is_risky_or_uncertain)
        results.append(FramePrior(frame_index=score.frame_index, target_histogram=target, mask=mask))
    return results
