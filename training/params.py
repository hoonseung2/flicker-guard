"""Samples synthesis parameters guaranteed to exceed a ThresholdProfile's
thresholds with a comfortable margin, so Detector re-validation in
training.synth is a confirmation step rather than blind trial and error.

Per-profile, not global (spec Global Constraint): every value here is
computed from the specific ThresholdProfile passed in, never from a fixed
min/max spanning all profiles.

Frequency note: `ThresholdProfile.max_flashes_per_second` is compared
against Detector's `flagged_frame_count_last_second` (flagged *frames* in a
trailing one-second window), not the visual flash rate (see
detector/scoring.py). `period_frames` here is chosen so the number of
flagged frames inside any trailing one-second window exceeds
`max_flashes_per_second` with margin.

Every target here is clamped (`_MAX_AREA_RATIO`, `_MAX_SATURATION_TARGET`,
`bright_target <= 1.0`), so a profile sitting close enough to one of those
caps could silently receive a target that does not actually exceed its own
threshold — whose only symptom would be every synthesis attempt exhausting
its retries with no diagnostic. `_require_exceeds` turns that into an
immediate, legible configuration error instead.
"""
from dataclasses import dataclass

import numpy as np

from detector.profiles import ThresholdProfile

_AREA_MARGIN = 0.10
_MAX_AREA_RATIO = 0.90
_RATIO_MARGIN = 0.05
_MAX_SATURATION_TARGET = 0.99


class ClipTooShortError(ValueError):
    """The clip has fewer frames than one injection window needs.

    A distinct type (rather than a bare ValueError) so callers can tell "this
    particular clip is unusable, skip it and carry on" apart from "this
    profile can never be satisfied", which is a fail-fast config error.
    """


@dataclass
class InjectionWindow:
    start_frame: int
    end_frame: int  # inclusive
    mask_top: int
    mask_left: int
    mask_height: int
    mask_width: int
    period_frames: int  # frames per alternation half-cycle (state held this long)
    ramp_frames: int  # Detector's trailing-window length (round(fps)) at sample time


@dataclass
class SynthParams:
    pattern: str  # "general" | "red"
    window: InjectionWindow
    dark_target: float | None = None
    bright_target: float | None = None
    red_rgb: tuple[float, float, float] | None = None
    baseline_rgb: tuple[float, float, float] | None = None


def _require_exceeds(
    achieved: float,
    threshold: float,
    profile_name: str,
    threshold_field: str,
    quantity: str,
    cap_reason: str,
) -> None:
    """Fail loudly when a clamp/rounding cap has eaten the safety margin.

    Every target here is capped (`_MAX_AREA_RATIO`, `_MAX_SATURATION_TARGET`,
    `bright_target <= 1.0`). A profile close enough to a cap silently stops
    getting a target that actually exceeds its own threshold, and the only
    symptom would be every synthesis attempt exhausting its retries with no
    diagnostic. Turn that into an immediate configuration error instead.
    """
    if achieved > threshold:
        return
    raise ValueError(
        f"profile {profile_name!r}: {quantity} {achieved:.4f} does not exceed "
        f"{threshold_field}={threshold:.4f} — {cap_reason}. This profile "
        f"cannot be synthesized against; widen the cap or relax the profile."
    )


def _mask_dims(frame_height: int, frame_width: int, target_area_ratio: float) -> tuple[int, int, int, int]:
    target_area_ratio = min(target_area_ratio, _MAX_AREA_RATIO)
    side_ratio = target_area_ratio ** 0.5
    mask_height = max(1, min(frame_height, round(frame_height * side_ratio)))
    mask_width = max(1, min(frame_width, round(frame_width * side_ratio)))
    mask_top = (frame_height - mask_height) // 2
    mask_left = (frame_width - mask_width) // 2
    return mask_top, mask_left, mask_height, mask_width


def _period_frames(profile: ThresholdProfile, fps: float) -> int:
    window_frames = max(1, round(fps))
    # Halve the just-crossing period for margin, so the steady-state flagged
    # count comfortably exceeds max_flashes_per_second rather than sitting
    # right at the boundary.
    period = window_frames / (2 * (profile.max_flashes_per_second + 1))
    return max(1, int(period))


def sample_synthesis_params(
    profile: ThresholdProfile,
    pattern: str,
    clip_frame_count: int,
    frame_height: int,
    frame_width: int,
    fps: float,
    rng: np.random.Generator,
) -> SynthParams:
    if pattern not in ("general", "red"):
        raise ValueError(f"unknown pattern: {pattern!r}")

    window_frames = max(1, round(fps))
    period_frames = _period_frames(profile, fps)
    # window_frames of ramp-up (counter filling) + 4 periods of guaranteed
    # steady-state tail once the trailing window is fully inside the pattern.
    duration_frames = window_frames + 4 * period_frames

    if clip_frame_count < duration_frames + 1:
        raise ClipTooShortError(
            f"clip has {clip_frame_count} frames, need at least "
            f"{duration_frames + 1} for fps={fps} and profile {profile.name!r}"
        )

    latest_start = clip_frame_count - duration_frames
    # PriorCalc's TargetHistogramSmoother needs a stretch of genuinely clean
    # (non-risky, non-uncertain) frames before the injected segment to ever
    # compute a non-None target histogram. Measured on the real 170-sample
    # dataset: samples whose injected segment started early (median start
    # ~15% into the clip) contributed zero training examples, while samples
    # that started around the clip's middle (median ~48%) contributed dozens
    # each -- see docs/superpowers/specs/2026-08-04-mask-diversification-design.md
    # section 3. When a clip is too short to offer the full desired runway,
    # give it as much as the clip can provide rather than rejecting it
    # outright (ClipTooShortError above already handles clips too short for
    # the injection window itself).
    desired_runway = 2 * window_frames
    min_start = min(desired_runway, latest_start)
    start_frame = int(rng.integers(min_start, latest_start + 1))
    end_frame = start_frame + duration_frames - 1

    mask_top, mask_left, mask_height, mask_width = _mask_dims(
        frame_height, frame_width, profile.max_area_ratio + _AREA_MARGIN
    )
    achieved_area_ratio = (mask_height * mask_width) / (frame_height * frame_width)
    _require_exceeds(
        achieved_area_ratio, profile.max_area_ratio, profile.name,
        "max_area_ratio", "injected mask area ratio",
        f"the mask is capped at {_MAX_AREA_RATIO:.2f} of the frame "
        f"(_MAX_AREA_RATIO) and rounded to whole pixels on "
        f"{frame_height}x{frame_width}",
    )

    window = InjectionWindow(
        start_frame=start_frame,
        end_frame=end_frame,
        mask_top=mask_top,
        mask_left=mask_left,
        mask_height=mask_height,
        mask_width=mask_width,
        period_frames=period_frames,
        ramp_frames=window_frames,
    )

    if pattern == "general":
        dark_target = profile.general_flash_dark_threshold * 0.5
        bright_target = min(1.0, dark_target + profile.general_flash_delta_threshold + 0.1)
        _require_exceeds(
            bright_target - dark_target, profile.general_flash_delta_threshold, profile.name,
            "general_flash_delta_threshold", "injected luminance delta",
            "bright_target is clamped to 1.0",
        )
        return SynthParams(pattern="general", window=window, dark_target=dark_target, bright_target=bright_target)

    target_ratio = min(_MAX_SATURATION_TARGET, profile.red_saturation_ratio_threshold + _RATIO_MARGIN)
    _require_exceeds(
        target_ratio, profile.red_saturation_ratio_threshold, profile.name,
        "red_saturation_ratio_threshold", "injected red saturation ratio",
        f"the target is capped at {_MAX_SATURATION_TARGET} (_MAX_SATURATION_TARGET)",
    )
    r = 0.9
    gb_sum = r / target_ratio - r
    red_rgb = (r, gb_sum / 2, gb_sum / 2)
    baseline_rgb = (0.3, 0.3, 0.3)
    return SynthParams(pattern="red", window=window, red_rgb=red_rgb, baseline_rgb=baseline_rgb)
