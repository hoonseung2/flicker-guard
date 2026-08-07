"""Tier 0 end to end: pick a strength per risk segment, apply it with a
fade, ask the Detector, raise the strength on whatever still fails.

Only risk segments are touched. Compressing the whole clip would be simpler
and is what the reference blind-deflickering implementation does -- and it
costs 21-35% of the mean luminance across the entire video, including the
parts that were never dangerous. Leaving safe frames untouched is the one
place this pipeline can be better than that, so it is worth the fade
machinery needed to re-enter a segment without a visible step.
"""
from dataclasses import dataclass

import numpy as np

from detector.luminance import srgb_to_linear
from detector.pipeline import run_detection
from detector.profiles import ThresholdProfile
from detector.segments import RiskSegment
from fallback.strength import required_strength
from fallback.transfer import compress_contrast, trailing_reference


@dataclass
class SegmentOutcome:
    start_frame: int
    end_frame: int
    initial_strength: float
    final_strength: float
    rounds: int
    passed: bool


@dataclass
class FallbackReport:
    segments: list[SegmentOutcome]
    passed: bool
    rounds: int
    remaining_segments: int


def fade_weights(length: int, margin_frames: int) -> np.ndarray:
    """Raised-cosine ramp in, flat, ramp out -- one multiplier per frame.

    Without it the strength steps from 0 to its full value at the segment
    boundary, and that step is itself a luminance transition the Detector
    can flag -- the correction would create the hazard it exists to remove.
    The ramp lives in the +-0.5s margin `run_detection` already pads each
    segment with, which is outside the risky frames proper.
    """
    weights = np.ones(length, dtype=np.float32)
    fade = min(margin_frames, length // 2)
    if fade <= 0:
        return weights

    steps = (np.arange(fade, dtype=np.float32) + 1.0) / (fade + 1.0)
    ramp = (0.5 * (1.0 - np.cos(np.pi * steps))).astype(np.float32)
    weights[:fade] = ramp
    weights[length - fade :] = ramp[::-1]
    return weights


def mitigate_with_fallback(
    frames: list[np.ndarray],
    fps: float,
    profile: ThresholdProfile,
    max_rounds: int = 6,
    strength_step: float = 0.1,
) -> tuple[list[np.ndarray], FallbackReport]:
    """Placeholder so `test_apply.py`'s module-level import succeeds ahead of
    the escalation loop, implemented in the following commit (Task 4 Step 8).
    Not exercised by any test in this commit."""
    raise NotImplementedError
