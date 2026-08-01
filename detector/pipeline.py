"""Wires the luminance / motion / flash / scoring / segment modules into a
single call that scans a full frame sequence."""
import numpy as np

from detector.flash import red_flash_mask, transition_mask
from detector.luminance import relative_luminance
from detector.motion import compensate_shift, estimate_global_shift
from detector.profiles import ThresholdProfile
from detector.scoring import FlickerScore, WindowedFlashCounter
from detector.segments import RiskSegment, scores_to_segments


def run_detection(
    frames: list[np.ndarray],
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
            mask = np.zeros(curr_luminance.shape, dtype=bool)
        else:
            dx, dy = estimate_global_shift(prev_luminance, curr_luminance)
            aligned_prev_luminance = compensate_shift(prev_luminance, -dx, -dy)
            aligned_prev_rgb = compensate_shift(prev_rgb, -dx, -dy)
            mask = transition_mask(aligned_prev_luminance, curr_luminance) | red_flash_mask(aligned_prev_rgb, frame)
        scores.append(counter.update(i, mask))
        prev_rgb, prev_luminance = frame, curr_luminance

    margin_frames = round(margin_seconds * fps)
    segments = scores_to_segments(scores, profile, margin_frames, total_frames=len(frames))
    return scores, segments
