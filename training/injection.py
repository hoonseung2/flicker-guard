"""Pixel-space PSE flicker injection. Alternates a rectangular region of a
clip between two fixed states every `window.period_frames` frames,
guaranteeing a transition Detector's own transition_mask/red_flash_mask will
flag (see training.params for how targets are chosen relative to a
profile).

Frames outside the injection window, and pixels outside the mask, are left
byte-identical to the input -- this is what makes the input clip usable
directly as Mitigator's clean ground truth outside the injected region.
"""
import numpy as np

from training.params import InjectionWindow

# WCAG/BT.709 sRGB encode, the algebraic inverse of detector.luminance's
# decode for the gray case (R=G=B=c): since the BT.709 coefficients sum to
# 1.0, relative_luminance(c, c, c) == decode(c) exactly.
_SRGB_A = 0.055
_LINEAR_CUTOFF = 0.0031308  # 0.04045 / 12.92


def _gray_channel_for_luminance(target_luminance: float) -> float:
    target_luminance = float(np.clip(target_luminance, 0.0, 1.0))
    if target_luminance <= _LINEAR_CUTOFF:
        return target_luminance * 12.92
    return (1 + _SRGB_A) * (target_luminance ** (1 / 2.4)) - _SRGB_A


def _is_pulse_frame(offset: int, period_frames: int) -> bool:
    return (offset // period_frames) % 2 == 1


def _mask_slice(window: InjectionWindow) -> tuple[slice, slice]:
    return (
        slice(window.mask_top, window.mask_top + window.mask_height),
        slice(window.mask_left, window.mask_left + window.mask_width),
    )


def inject_general_flash(
    frames: list[np.ndarray],
    window: InjectionWindow,
    dark_target: float,
    bright_target: float,
) -> list[np.ndarray]:
    dark_channel = _gray_channel_for_luminance(dark_target)
    bright_channel = _gray_channel_for_luminance(bright_target)
    rows, cols = _mask_slice(window)

    out = [frame.copy() for frame in frames]
    for i in range(window.start_frame, window.end_frame + 1):
        offset = i - window.start_frame
        channel = bright_channel if _is_pulse_frame(offset, window.period_frames) else dark_channel
        out[i][rows, cols, :] = channel
    return out


def inject_red_flash(
    frames: list[np.ndarray],
    window: InjectionWindow,
    red_rgb: tuple[float, float, float],
    baseline_rgb: tuple[float, float, float],
) -> list[np.ndarray]:
    rows, cols = _mask_slice(window)

    out = [frame.copy() for frame in frames]
    for i in range(window.start_frame, window.end_frame + 1):
        offset = i - window.start_frame
        color = red_rgb if _is_pulse_frame(offset, window.period_frames) else baseline_rgb
        out[i][rows, cols, 0] = color[0]
        out[i][rows, cols, 1] = color[1]
        out[i][rows, cols, 2] = color[2]
    return out
