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
"""
from dataclasses import dataclass

import numpy as np

from detector.profiles import ThresholdProfile

_AREA_MARGIN = 0.10
_MAX_AREA_RATIO = 0.90
_RATIO_MARGIN = 0.05
_MAX_SATURATION_TARGET = 0.99


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
        raise ValueError(
            f"clip has {clip_frame_count} frames, need at least "
            f"{duration_frames + 1} for fps={fps} and profile {profile.name!r}"
        )

    latest_start = clip_frame_count - duration_frames
    start_frame = int(rng.integers(1, latest_start + 1))
    end_frame = start_frame + duration_frames - 1

    mask_top, mask_left, mask_height, mask_width = _mask_dims(
        frame_height, frame_width, profile.max_area_ratio + _AREA_MARGIN
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
        return SynthParams(pattern="general", window=window, dark_target=dark_target, bright_target=bright_target)

    target_ratio = min(_MAX_SATURATION_TARGET, profile.red_saturation_ratio_threshold + _RATIO_MARGIN)
    r = 0.9
    gb_sum = r / target_ratio - r
    red_rgb = (r, gb_sum / 2, gb_sum / 2)
    baseline_rgb = (0.3, 0.3, 0.3)
    return SynthParams(pattern="red", window=window, red_rgb=red_rgb, baseline_rgb=baseline_rgb)
