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


def _apply_segments(
    frames: list[np.ndarray],
    frames_linear: list[np.ndarray],
    segments: list[RiskSegment],
    strengths: dict[tuple[int, int], float],
    window_frames: int,
    margin_frames: int,
) -> list[np.ndarray]:
    """Rebuild the clip with each segment compressed at its own strength.

    Frames outside every segment are passed through by identity (the same
    array object), not recomputed at strength 0 -- an sRGB round trip is
    lossy at float32 and would leave a measurable difference on frames the
    Detector already called safe.
    """
    output = list(frames)
    total_frames = len(frames)
    for segment in segments:
        length = segment.end_frame - segment.start_frame + 1
        # `fade_weights(length, margin_frames)` assumes a full `margin_frames`
        # of genuinely safe buffer sits on each side of the segment. That is
        # true whenever `scores_to_segments`'s padding was NOT clipped by the
        # clip's own boundary -- but when a segment's start/end already sits
        # at frame 0 or the last frame, the clamp (`max(0, ...)` /
        # `min(total-1, ...)`) means the real safe buffer on that side is
        # only the distance to the clip edge, sometimes zero. Fading over
        # frames that are still genuinely risky caps the achievable effective
        # strength at `weights[offset]`, a ceiling no amount of escalating
        # `strength` can cross.
        start_margin = min(margin_frames, segment.start_frame)
        end_margin = min(margin_frames, total_frames - 1 - segment.end_frame)
        weights = _segment_weights(length, start_margin, end_margin)
        strength = strengths[(segment.start_frame, segment.end_frame)]
        for offset in range(length):
            index = segment.start_frame + offset
            reference = trailing_reference(frames_linear, index, window_frames)
            output[index] = compress_contrast(
                frames[index], reference, float(strength * weights[offset])
            )
    return output


def _segment_weights(length: int, start_margin: int, end_margin: int) -> np.ndarray:
    """Like `fade_weights`, but the entry and exit ramp lengths are set
    independently instead of both being `margin_frames`.

    Reuses `fade_weights` itself for the ramp shape: `fade_weights(length,
    start_margin)`'s first `min(start_margin, length // 2)` entries are
    exactly the entry ramp for that margin, and symmetrically for the exit
    ramp from `fade_weights(length, end_margin)`. Each call independently
    caps its ramp at `length // 2`, so the two slices taken together can
    never overlap.
    """
    entry = fade_weights(length, start_margin)
    exit_ = fade_weights(length, end_margin)
    start_fade = min(start_margin, length // 2)
    end_fade = min(end_margin, length // 2)
    weights = np.ones(length, dtype=np.float32)
    if start_fade > 0:
        weights[:start_fade] = entry[:start_fade]
    if end_fade > 0:
        weights[length - end_fade :] = exit_[length - end_fade :]
    return weights


def mitigate_with_fallback(
    frames: list[np.ndarray],
    fps: float,
    profile: ThresholdProfile,
    max_rounds: int = 6,
    strength_step: float = 0.1,
) -> tuple[list[np.ndarray], FallbackReport]:
    """Compress every risk segment until the Detector passes the whole clip.

    Verification runs over the whole clip every round, never over a segment
    in isolation: `WindowedFlashCounter` primes its window with flagged
    slots and marks frame 0 uncertain, so a segment measured on its own is
    judged under different conditions than the one that decides pass or
    fail.
    """
    _scores, segments = run_detection(frames, fps=fps, profile=profile)
    if not segments:
        return list(frames), FallbackReport(
            segments=[], passed=True, rounds=0, remaining_segments=0
        )

    window_frames = max(1, round(fps))
    margin_frames = round(0.5 * fps)

    keys = [(segment.start_frame, segment.end_frame) for segment in segments]
    initial = {
        key: required_strength(frames[key[0] : key[1] + 1], profile) for key in keys
    }
    # A segment the analytic estimate rates as needing nothing still has to
    # move: run_detection flagged it, so something the estimate cannot see
    # (the red-flash XOR, the dark-threshold condition, the windowed area
    # maximum) is driving it. Starting at exactly 0 would waste a round.
    strengths = {key: max(value, strength_step) for key, value in initial.items()}
    rounds_used = {key: 0 for key in keys}

    frames_linear = [srgb_to_linear(frame.astype(np.float32)) for frame in frames]

    output = list(frames)
    remaining: list[RiskSegment] = []
    passed = False
    rounds = 0

    for rounds in range(1, max_rounds + 1):
        output = _apply_segments(
            frames, frames_linear, segments, strengths, window_frames, margin_frames
        )
        _scores, remaining = run_detection(output, fps=fps, profile=profile)
        if not remaining:
            passed = True
            break

        risky_frames = set()
        for segment in remaining:
            risky_frames.update(range(segment.start_frame, segment.end_frame + 1))

        for segment, key in zip(segments, keys):
            overlaps = risky_frames.intersection(
                range(segment.start_frame, segment.end_frame + 1)
            )
            if overlaps:
                strengths[key] = min(1.0, strengths[key] + strength_step)
                rounds_used[key] += 1

    still_risky = set()
    for segment in remaining:
        still_risky.update(range(segment.start_frame, segment.end_frame + 1))

    outcomes = [
        SegmentOutcome(
            start_frame=key[0],
            end_frame=key[1],
            initial_strength=initial[key],
            final_strength=strengths[key],
            rounds=rounds_used[key],
            passed=not still_risky.intersection(range(key[0], key[1] + 1)),
        )
        for key in keys
    ]

    return output, FallbackReport(
        segments=outcomes,
        passed=passed,
        rounds=rounds,
        remaining_segments=len(remaining),
    )
