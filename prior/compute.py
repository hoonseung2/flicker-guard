"""Wires Detector's extended run_detection_with_masks together with
prior.histogram: for every frame, produces a (smoothed target illumination
histogram, per-pixel risk mask) pair for Mitigator conditioning.

"Risky" for smoothing purposes uses Detector's *margin-padded* segments
(default margin_seconds=0.5, i.e. run_detection_with_masks's default) rather
than the zero-margin check DatasetSynth's validation uses -- Mitigator will
only ever receive margin-padded segments from BufferManager in production, so
PriorCalc's notion of "risky" should match what Mitigator actually sees.
"""
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from detector.luminance import relative_luminance
from detector.pipeline import run_detection_with_masks
from detector.profiles import ThresholdProfile
from prior.histogram import TargetHistogramSmoother, compute_illumination_histogram

# Canonical histogram bin count -- PriorCalc owns histogram computation, so
# other modules that need this number (e.g. Mitigator's histogram-embedding
# input width) should import it from here rather than hardcoding a
# duplicate literal that can silently drift out of sync.
DEFAULT_N_BINS = 64


@dataclass
class FramePrior:
    frame_index: int
    # `target_histogram` is None whenever no clean (non-risky, non-uncertain)
    # frame has been seen yet -- there is nothing to average and nothing to
    # fall back to. This is the ROUTINE, EXPECTED case for a clip's opening
    # stretch, not an error path:
    #
    # - Detector's WindowedFlashCounter needs a full trailing second before it
    #   stops reporting `uncertain`, so *every* clip has None targets for at
    #   least its first `round(fps)` frames -- on a clip shorter than one
    #   second, that is every frame.
    # - Any additional prefix before the first genuinely clean frame is ever
    #   seen is also None. Measured on an 80-frame 10 fps clip with a hazard
    #   injected at frames 10-20, i.e. starting right as the priming window
    #   ends: frames 0-31 were None, not just 0-9.
    #
    # Callers (Mitigator) must treat None as normal input and handle it
    # explicitly -- skip conditioning, hold the previous target, or exclude the
    # frame from training -- rather than as an exceptional/failed computation.
    target_histogram: np.ndarray | None
    mask: np.ndarray


def compute_prior(
    frames: Iterable[np.ndarray],
    fps: float,
    profile: ThresholdProfile,
    n_bins: int = DEFAULT_N_BINS,
) -> list[FramePrior]:
    # Materialise first: unlike run_detection*, compute_prior needs `frames`
    # twice (once for detection, once to re-derive each frame's histogram).
    # A one-shot iterable -- e.g. detector.cli.read_video_frames, which
    # returns a generator -- would be drained by run_detection_with_masks and
    # the later zip would then silently yield an empty result.
    frames = list(frames)
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
