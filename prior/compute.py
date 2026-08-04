"""Wires Detector's extended run_detection_with_masks together with
prior.histogram: for every frame, produces a (target illumination
histogram, per-pixel risk mask) pair for Mitigator conditioning.

Target histograms are derived from the CURRENT frame's own non-masked
pixels, not a trailing window of past clean frames. The old approach
(a smoothed average of recent temporally-clean frames) starves entirely on
footage where a margin-padded risky segment runs for many consecutive
frames with no clean frame anywhere in it -- a real continuous strobe clip
never lets up. Measured on a real concert-lighting clip: 418 consecutive
risky frames with zero clean frames after the first ~90, all sharing one
stale target frozen from before the strobing ever started, which is why
the trained model's correction barely moved the needle on that clip.

A per-pixel risk mask is only ever the SPATIAL extent of an actual detected
transition (see detector.pipeline's transition_mask / red_flash_mask), so
the non-masked pixels of that same frame are a valid "what does
non-flickering illumination look like right now" sample whenever the mask
doesn't cover (nearly) the whole frame -- true for the overwhelming
majority of real footage, since ThresholdProfile.max_area_ratio caps how
much area a compliant flash pattern can legitimately cover. When a frame's
mask does cover nearly everything (or detection is still uncertain, which
flags the whole frame conservatively -- see detector/pipeline.py's I4
comment), there is no trustworthy in-frame reference, so the last
trustworthy target is held instead, same fallback semantics as before.
"""
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from detector.luminance import relative_luminance
from detector.pipeline import run_detection_with_masks
from detector.profiles import ThresholdProfile
from prior.histogram import compute_illumination_histogram

# Canonical histogram bin count -- PriorCalc owns histogram computation, so
# other modules that need this number (e.g. Mitigator's histogram-embedding
# input width) should import it from here rather than hardcoding a
# duplicate literal that can silently drift out of sync.
DEFAULT_N_BINS = 64

# Below this fraction of non-masked frame area, the reference sample is too
# small (and too likely biased toward whatever narrow sliver is left) to
# trust as "what normal illumination looks like" -- hold the last
# trustworthy target instead of computing a shaky one from a handful of
# pixels.
_MIN_REFERENCE_PIXEL_RATIO = 0.05


@dataclass
class FramePrior:
    frame_index: int
    # `target_histogram` is None until the first frame with enough
    # non-masked area is seen -- there is nothing to derive a reference
    # from and nothing to fall back to yet. This is the ROUTINE, EXPECTED
    # case for a clip's opening stretch (Detector's WindowedFlashCounter
    # reports every frame uncertain, and uncertain frames are conservatively
    # masked in full -- see detector/pipeline.py) not an error path.
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
    scores, _segments, masks = run_detection_with_masks(frames, fps=fps, profile=profile)

    results: list[FramePrior] = []
    last_valid_target: np.ndarray | None = None
    for frame, score, mask in zip(frames, scores, masks):
        non_mask_pixels = relative_luminance(frame)[~mask]
        if non_mask_pixels.size >= _MIN_REFERENCE_PIXEL_RATIO * mask.size:
            last_valid_target = compute_illumination_histogram(non_mask_pixels, n_bins=n_bins)
        results.append(FramePrior(frame_index=score.frame_index, target_histogram=last_valid_target, mask=mask))
    return results
