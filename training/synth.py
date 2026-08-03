"""Orchestrates one synthesis attempt: sample parameters, inject the
pattern, and re-validate with the real Detector pipeline before accepting.
Rejected attempts are retried with fresh parameters up to `max_retries`
times -- this is the "confirmation, not blind generation" loop the spec
calls for."""
from dataclasses import dataclass

import numpy as np

from detector.pipeline import run_detection
from detector.profiles import ThresholdProfile
from detector.segments import RiskSegment
from training.injection import (
    inject_general_flash,
    inject_general_flash_realistic,
    inject_red_flash,
    inject_red_flash_realistic,
)
from training.params import InjectionWindow, SynthParams, sample_synthesis_params


@dataclass
class SynthesizedSample:
    pattern: str
    profile_name: str
    window: InjectionWindow
    # Intentionally an alias of the caller's frame list/arrays, never a copy:
    # the batch loop reuses one clip across 12 profile x pattern combos, so
    # copying here would multiply peak memory for no benefit. Nothing in the
    # pipeline mutates it (injection.py copies before writing). Never mutate.
    clean_frames: list[np.ndarray]
    degraded_frames: list[np.ndarray]
    segments: list[RiskSegment]


def _window_is_covered(window: InjectionWindow, segments: list[RiskSegment]) -> bool:
    core_start = window.start_frame + window.ramp_frames
    return any(
        segment.start_frame <= core_start and segment.end_frame >= window.end_frame
        for segment in segments
    )


def _apply_pattern(clean_frames: list[np.ndarray], params: SynthParams) -> list[np.ndarray]:
    if params.pattern == "general":
        return inject_general_flash(clean_frames, params.window, params.dark_target, params.bright_target)
    return inject_red_flash(clean_frames, params.window, params.red_rgb, params.baseline_rgb)


def synthesize_sample(
    clean_frames: list[np.ndarray],
    fps: float,
    profile: ThresholdProfile,
    pattern: str,
    rng: np.random.Generator,
    max_retries: int = 5,
) -> SynthesizedSample | None:
    frame_height, frame_width = clean_frames[0].shape[:2]

    for _ in range(max_retries):
        params = sample_synthesis_params(
            profile, pattern, len(clean_frames), frame_height, frame_width, fps, rng
        )
        degraded_frames = _apply_pattern(clean_frames, params)
        _, segments = run_detection(degraded_frames, fps=fps, profile=profile, margin_seconds=0.0)
        if _window_is_covered(params.window, segments):
            return SynthesizedSample(
                pattern=pattern,
                profile_name=profile.name,
                window=params.window,
                clean_frames=clean_frames,  # intentional alias, not a copy — see dataclass field
                degraded_frames=degraded_frames,
                segments=segments,
            )
    return None


def _apply_pattern_realistic(clean_frames: list[np.ndarray], params: SynthParams) -> list[np.ndarray]:
    if params.pattern == "general":
        return inject_general_flash_realistic(clean_frames, params.window)
    return inject_red_flash_realistic(clean_frames, params.window)


def synthesize_sample_realistic(
    clean_frames: list[np.ndarray],
    fps: float,
    profile: ThresholdProfile,
    pattern: str,
    rng: np.random.Generator,
    max_retries: int = 5,
) -> SynthesizedSample | None:
    """Same retry-and-validate structure as synthesize_sample, but injects
    with the linear-space gain functions instead of the flat-color-overwrite
    ones. sample_synthesis_params is reused only for its InjectionWindow
    (position/size/timing) -- its dark_target/bright_target/red_rgb/
    baseline_rgb fields are computed but unused here, since the realistic
    path uses fixed gain constants (see training.injection) rather than
    per-profile absolute targets."""
    frame_height, frame_width = clean_frames[0].shape[:2]

    for _ in range(max_retries):
        params = sample_synthesis_params(
            profile, pattern, len(clean_frames), frame_height, frame_width, fps, rng
        )
        degraded_frames = _apply_pattern_realistic(clean_frames, params)
        _, segments = run_detection(degraded_frames, fps=fps, profile=profile, margin_seconds=0.0)
        if _window_is_covered(params.window, segments):
            return SynthesizedSample(
                pattern=pattern,
                profile_name=profile.name,
                window=params.window,
                clean_frames=clean_frames,
                degraded_frames=degraded_frames,
                segments=segments,
            )
    return None
