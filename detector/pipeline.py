"""Wires the luminance / motion / flash / scoring / segment modules into a
single call that scans a full frame sequence.

**MVP limitations — external validation required.** This detector is a
simplified approximation of the WCAG / Harding General Flash and Red Flash
Threshold Algorithm, not an implementation of it:

- Flash counting is flagged-frame counting over a trailing one-second window,
  not full opposing-transition-pair analysis; the reported count runs at
  roughly 2x the true visual flash rate (conservative, see
  `detector/scoring.py`).
- `ThresholdProfile` encodes only a flash-frequency limit and a flagged-area
  limit. Ofcom's dark-scene sub-rule (luminance < 160 with contrast >= 20)
  and Japan's high-contrast pattern-density rule are **not** encoded and are
  therefore **not** detected (plan Task 5 deferred them to a future schema
  extension; recorded here per final-review finding I5).
- Motion compensation is global/pan-only; local object motion is not
  compensated and leaves a small residual flagged area at frame borders.

Per README section 9, Harding FPA or equivalent external validation is still
required before production use. Nothing here may be presented as a guarantee
that content is safe.

Streaming (final-review finding I6): `frames` is any iterable — the pipeline
is strictly causal and never holds more than the previous and current frame,
so a generator from `detector.cli.read_video_frames` streams a video without
materialising it in RAM.

Uncertain edges (final-review finding I4, README section 7): frame 0 has no
predecessor, so its transition is unknown rather than safe. It is fed to the
counter as a fully flagged, explicitly unmeasured frame, which can only push
the verdict toward "risky".
"""
from collections.abc import Iterable

import numpy as np

from detector.flash import red_flash_mask, transition_mask
from detector.luminance import relative_luminance
from detector.motion import compensate_shift, estimate_global_shift
from detector.profiles import ThresholdProfile
from detector.scoring import FlickerScore, WindowedFlashCounter
from detector.segments import RiskSegment, scores_to_segments


def run_detection(
    frames: Iterable[np.ndarray],
    fps: float,
    profile: ThresholdProfile,
    margin_seconds: float = 0.5,
) -> tuple[list[FlickerScore], list[RiskSegment]]:
    counter = WindowedFlashCounter(fps=fps)
    scores: list[FlickerScore] = []

    prev_rgb = None
    prev_luminance = None
    for i, frame in enumerate(frames):
        curr_luminance = relative_luminance(frame)
        if prev_rgb is None:
            # No predecessor: the transition into this frame is unknown, so
            # assume the worst instead of fabricating an all-safe mask (I4).
            mask = np.ones(curr_luminance.shape, dtype=bool)
            uncertain = True
        else:
            dx, dy = estimate_global_shift(prev_luminance, curr_luminance)
            aligned_prev_luminance = compensate_shift(prev_luminance, dx, dy)
            aligned_prev_rgb = compensate_shift(prev_rgb, dx, dy)
            mask = transition_mask(
                aligned_prev_luminance,
                curr_luminance,
                dark_threshold=profile.general_flash_dark_threshold,
                delta_threshold=profile.general_flash_delta_threshold,
            ) | red_flash_mask(
                aligned_prev_rgb,
                frame,
                saturation_ratio_threshold=profile.red_saturation_ratio_threshold,
            )
            uncertain = False
        scores.append(counter.update(i, mask, uncertain=uncertain))
        prev_rgb, prev_luminance = frame, curr_luminance

    margin_frames = round(margin_seconds * fps)
    segments = scores_to_segments(scores, profile, margin_frames, total_frames=len(scores))
    return scores, segments
