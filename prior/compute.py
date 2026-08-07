"""Wires Detector's extended run_detection_with_masks together with
prior.histogram: for every frame, produces a (target illumination
histogram, per-pixel risk mask) pair for Mitigator conditioning.

Target histograms are derived from a trailing-window average of the CURRENT
frame's own non-masked pixels, not a window of past temporally-clean
frames. The original approach (smoothed average of recent frames the
detector called entirely non-risky) starves on footage where a
margin-padded risky segment runs for many consecutive frames with no clean
frame anywhere in it -- a real continuous strobe clip never lets up.
Measured on a real concert-lighting clip: 418 consecutive risky frames with
zero clean frames after the first ~90, all sharing one target frozen from
before the strobing ever started.

A per-pixel risk mask is only ever the SPATIAL extent of an actual detected
transition (see detector.pipeline's transition_mask / red_flash_mask), so
the non-masked pixels of that same frame are a valid "what does
non-flickering illumination look like right now" sample whenever the mask
doesn't cover (nearly) the whole frame -- true for the overwhelming
majority of real footage, since ThresholdProfile.max_area_ratio caps how
much area a compliant flash pattern can legitimately cover. Averaging these
per-frame spatial samples over a trailing window (same length as the old
smoother's) keeps the target from jittering frame-to-frame the way a raw,
independently-recomputed-every-frame spatial sample would -- this is the
hybrid of the two approaches: spatial SOURCE (never starves), temporal
SMOOTHING (stays stable). When a frame's mask does cover nearly everything
(or detection is still uncertain, which flags the whole frame
conservatively -- see detector/pipeline.py's I4 comment), there is no
trustworthy in-frame reference for that frame; it simply does not
contribute to the trailing window, and the last computed average is held.
"""
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from detector.luminance import relative_luminance
from detector.pipeline import run_detection_with_masks
from detector.profiles import ThresholdProfile
from detector.segments import RiskSegment
# required_strength lives in fallback/ because it was written for the Tier 0
# fallback path, but it computes a measurement (how far a segment's deltas
# sit past the Detector's threshold), not a Tier 0 policy decision -- so
# PriorCalc reuses it here rather than duplicating the quantile logic.
from fallback.strength import required_strength
from prior.histogram import compute_illumination_histogram

# Canonical histogram bin count -- PriorCalc owns histogram computation, so
# other modules that need this number (e.g. Mitigator's histogram-embedding
# input width) should import it from here rather than hardcoding a
# duplicate literal that can silently drift out of sync.
DEFAULT_N_BINS = 64

# Below this fraction of non-masked frame area, the reference sample is too
# small (and too likely biased toward whatever narrow sliver is left) to
# trust as "what normal illumination looks like" -- skip it rather than
# feeding a shaky sample from a handful of pixels into the trailing average.
_MIN_REFERENCE_PIXEL_RATIO = 0.05

# How long a flagged pixel keeps counting as "needs repair" after the
# transition that flagged it.
#
# Detector's per-frame mask marks pixels that CHANGED since the previous
# frame. That is the right question for "is this content hazardous" -- the
# standards count transitions -- but the wrong one for "what do I repair":
# while a flash holds its bright state there is no transition, so the mask
# empties even though those pixels are still wrong. Measured across 420
# frames of the real training set, the raw transition mask covered 92.1%
# (median) of the actually-damaged pixels on transition frames but only
# 10.4% on hold frames, and hold frames were 69% of the total.
#
# OR-ing the recent masks together restores that coverage: at 0.2s the
# median coverage of genuinely damaged pixels rose from 15.7% to ~96%, and
# the trained model's median PSNR gain rose from +0.23 dB to +1.76 dB
# (upper bound with a ground-truth damage mask: +4.20 dB). Longer is not
# strictly better -- the mask is also a model input, so an over-wide mask
# tells the network that clean pixels are damaged and it degrades them.
_DEFAULT_MASK_PERSISTENCE_SECONDS = 0.2


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
    # How far this frame's risk segment has to be pushed, as a fraction of
    # its per-pixel luminance deltas -- 0.0 outside any segment.
    #
    # The model sees a three-frame window and cannot tell from it how severe
    # the surrounding segment is. The old conditioning was a histogram of
    # "what normal illumination looks like", which describes a destination
    # but not a distance; the trained model reached a median 14.6% of it.
    # This is the distance, derived from the same quantity the Detector
    # thresholds on. Apple's published mitigation scales its strength the
    # same way, from a measured risk score rather than a fixed constant.
    required_strength: float = 0.0


def _assign_segment_strengths(
    frames: list[np.ndarray],
    priors: list[FramePrior],
    segments: list[RiskSegment],
    profile: ThresholdProfile,
) -> None:
    """Give every frame of a risk segment that segment's required strength.

    The strength is a per-segment quantity by construction --
    `fallback.strength.required_strength` takes the maximum over the
    segment's frame pairs, because a segment is flagged if any pair in it
    trips the area limit. Handing different frames of one segment different
    values would be incoherent, so they all get the segment's value.
    """
    for segment in segments:
        window = frames[segment.start_frame : segment.end_frame + 1]
        strength = required_strength(window, profile)
        for index in range(segment.start_frame, segment.end_frame + 1):
            priors[index].required_strength = strength


def compute_prior(
    frames: Iterable[np.ndarray],
    fps: float,
    profile: ThresholdProfile,
    n_bins: int = DEFAULT_N_BINS,
    mask_persistence_seconds: float = _DEFAULT_MASK_PERSISTENCE_SECONDS,
) -> list[FramePrior]:
    # Materialise first: unlike run_detection*, compute_prior needs `frames`
    # twice (once for detection, once to re-derive each frame's histogram).
    # A one-shot iterable -- e.g. detector.cli.read_video_frames, which
    # returns a generator -- would be drained by run_detection_with_masks and
    # the later zip would then silently yield an empty result.
    frames = list(frames)
    scores, segments, masks = run_detection_with_masks(frames, fps=fps, profile=profile)

    # Same trailing-window length the old temporal smoother used -- one
    # second's worth of frames is long enough to damp frame-to-frame
    # jitter without holding on to samples so old they no longer reflect
    # the current scene.
    window_frames = max(1, round(fps))
    window: list[np.ndarray] = []

    # 0 seconds disables persistence entirely, leaving the Detector's own
    # transition mask untouched.
    persistence_frames = max(1, round(mask_persistence_seconds * fps)) if mask_persistence_seconds > 0 else 1
    recent_masks: deque[np.ndarray] = deque(maxlen=persistence_frames)

    results: list[FramePrior] = []
    last_valid_target: np.ndarray | None = None
    for frame, score, raw_mask in zip(frames, scores, masks):
        recent_masks.append(raw_mask)
        mask = np.logical_or.reduce(recent_masks) if len(recent_masks) > 1 else raw_mask
        non_mask_pixels = relative_luminance(frame)[~mask]
        if non_mask_pixels.size >= _MIN_REFERENCE_PIXEL_RATIO * mask.size:
            window.append(compute_illumination_histogram(non_mask_pixels, n_bins=n_bins))
            if len(window) > window_frames:
                window.pop(0)
            last_valid_target = np.mean(window, axis=0)
        results.append(FramePrior(frame_index=score.frame_index, target_histogram=last_valid_target, mask=mask))
    _assign_segment_strengths(frames, results, segments, profile)
    return results
