"""The Tier 0 correction itself: pull a frame's pixels toward a reference
colour, in linear light, by a strength in [0, 1].

Why compression and not mean-matching: scaling a frame so its *mean*
luminance matches a reference leaves the ratios inside the frame untouched,
so the per-pixel differences the Detector actually thresholds on survive
almost unchanged -- brightening the dark frames by as much as it darkens
the bright ones. An affine pull toward a single colour is what collapses
those per-pixel differences, which is why Apple's published Video Flashing
Reduction transfer function has the same affine shape.

Why linear light: the sRGB curve is non-linear, so an affine map applied to
encoded values scales each channel differently and tints the picture. In
linear light the same affine on all three channels gives exactly the same
affine on relative luminance, since luminance is a fixed weighted sum of
the linear channels. That equality is what lets strength.py compute the
required strength instead of searching for it.
"""
import numpy as np

from detector.luminance import linear_to_srgb, srgb_to_linear


def trailing_reference(
    frames_linear: list[np.ndarray], index: int, window_frames: int
) -> np.ndarray:
    """Per-channel mean over the window of `window_frames` frames ending at
    (and including) `index`, clamped at the start of the clip.

    Trailing rather than centred so the correction stays causal and can be
    lifted into a streaming path later without changing its output.
    """
    start = max(0, index - window_frames + 1)
    window = frames_linear[start : index + 1]
    return np.mean([frame.reshape(-1, 3).mean(axis=0) for frame in window], axis=0).astype(
        np.float32
    )


def compress_contrast(
    frame_rgb: np.ndarray, reference_linear: np.ndarray, strength: float
) -> np.ndarray:
    """Compress `frame_rgb` toward `reference_linear` by `strength`.

    `strength` 0 returns the frame unchanged; 1 returns a frame of the
    reference colour, which has no internal contrast and therefore no
    per-pixel difference from the next such frame.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")

    linear = srgb_to_linear(frame_rgb.astype(np.float32))
    compressed = reference_linear + (linear - reference_linear) * (1.0 - strength)
    return linear_to_srgb(np.clip(compressed, 0.0, 1.0)).astype(np.float32)
