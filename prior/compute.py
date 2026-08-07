"""Wires Detector's extended run_detection_with_masks into a per-frame
(risk mask, required strength) pair for Mitigator conditioning.

A per-pixel risk mask is only ever the SPATIAL extent of an actual detected
transition (see detector.pipeline's transition_mask / red_flash_mask). The
mask alone does not say how severe the surrounding risk segment is, which
is what `required_strength` (below) is for.

This module previously also derived a per-frame target illumination
histogram (a trailing-window average of the frame's own non-masked pixels)
for the Mitigator to condition on directly. That path is retired: the
self-supervised objective (training.mitigator_losses.self_supervised_loss)
does not grade against a target distribution, only `required_strength` and
the mask.
"""
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from detector.pipeline import run_detection_with_masks
from detector.profiles import ThresholdProfile
from detector.segments import RiskSegment
# required_strength lives in fallback/ because it was written for the Tier 0
# fallback path, but it computes a measurement (how far a segment's deltas
# sit past the Detector's threshold), not a Tier 0 policy decision -- so
# PriorCalc reuses it here rather than duplicating the quantile logic.
from fallback.strength import required_strength

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
    mask: np.ndarray
    # How far this frame's risk segment has to be pushed, as a fraction of
    # its per-pixel luminance deltas -- 0.0 outside any segment.
    #
    # The model sees a three-frame window and cannot tell from it how severe
    # the surrounding segment is. The retired conditioning was a histogram of
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
    mask_persistence_seconds: float = _DEFAULT_MASK_PERSISTENCE_SECONDS,
) -> list[FramePrior]:
    # Materialise first: unlike run_detection*, compute_prior needs `frames`
    # twice (once for detection, once for _assign_segment_strengths below).
    # A one-shot iterable -- e.g. detector.cli.read_video_frames, which
    # returns a generator -- would be drained by run_detection_with_masks and
    # the later pass would then silently see an empty result.
    frames = list(frames)
    scores, segments, masks = run_detection_with_masks(frames, fps=fps, profile=profile)

    # 0 seconds disables persistence entirely, leaving the Detector's own
    # transition mask untouched.
    persistence_frames = max(1, round(mask_persistence_seconds * fps)) if mask_persistence_seconds > 0 else 1
    recent_masks: deque[np.ndarray] = deque(maxlen=persistence_frames)

    results: list[FramePrior] = []
    for score, raw_mask in zip(scores, masks):
        recent_masks.append(raw_mask)
        mask = np.logical_or.reduce(recent_masks) if len(recent_masks) > 1 else raw_mask
        results.append(FramePrior(frame_index=score.frame_index, mask=mask))
    _assign_segment_strengths(frames, results, segments, profile)
    return results
