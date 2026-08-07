"""How much compression this footage actually needs, computed rather than
searched.

`transfer.compress_contrast` scales the per-pixel luminance difference
between consecutive frames by exactly (1 - strength) as long as the
reference barely moves between them, which a one-second trailing mean does.
`detector.flash.transition_mask` flags a pixel when that difference exceeds
`general_flash_delta_threshold`, and `detector.segments._is_risky` fires
when the flagged fraction exceeds `max_area_ratio`. So the binding quantity
is the delta at the (1 - max_area_ratio) quantile: bring that to the
threshold and the flagged area lands at the limit.

The result is an estimate, not a verdict. Three things it cannot see:
`transition_mask` also requires the darker of the two frames to be below
`general_flash_dark_threshold`, and compression moves pixels across that
line in both directions; `flash.red_flash_mask` is a boolean XOR of
"is this pixel saturated red", which converges to zero as strength
approaches 1 but not proportionally to (1 - strength); and the flagged area
the profile is compared against is a maximum over a one-second window, not
a per-pair quantity. `apply.mitigate_with_fallback` therefore treats this
as a starting point and lets the real Detector decide.
"""
import numpy as np

from detector.luminance import relative_luminance
from detector.motion import compensate_shift, estimate_global_shift
from detector.profiles import ThresholdProfile


def motion_compensated_deltas(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Per-pixel |luminance change| between consecutive frames, after
    cancelling global camera motion.

    Mirrors `detector.pipeline._iter_scores_and_masks`: it compares
    `compensate_shift(prev)` against `curr`, so measuring without
    compensation would count a pan's edge displacement as a flash and
    demand compression that the Detector never asked for.
    """
    luminances = [relative_luminance(frame) for frame in frames]
    deltas = []
    for previous, current in zip(luminances[:-1], luminances[1:]):
        dx, dy = estimate_global_shift(previous, current)
        aligned_previous = compensate_shift(previous, dx, dy)
        deltas.append(np.abs(current - aligned_previous))
    return deltas


def required_strength(frames: list[np.ndarray], profile: ThresholdProfile) -> float:
    """Smallest strength that brings every frame pair's flagged area to
    `profile.max_area_ratio`, or 0.0 if none of them exceed it."""
    threshold = profile.general_flash_delta_threshold
    quantile = 1.0 - profile.max_area_ratio

    strongest = 0.0
    for delta in motion_compensated_deltas(frames):
        binding_delta = float(np.quantile(delta, quantile))
        if binding_delta > threshold:
            strongest = max(strongest, 1.0 - threshold / binding_delta)
    return float(min(1.0, strongest))
