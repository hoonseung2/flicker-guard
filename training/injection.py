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


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Array version of the sRGB decode -- the same formula
    `_gray_channel_for_luminance` inverts for the scalar/gray case, kept as a
    separate function (not shared with it) so neither has to change to
    support the other."""
    low = c <= 0.04045
    return np.where(low, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    low = c <= _LINEAR_CUTOFF
    return np.where(low, c * 12.92, (1 + _SRGB_A) * (c ** (1 / 2.4)) - _SRGB_A)


def inject_general_flash_realistic(
    frames: list[np.ndarray],
    window: InjectionWindow,
    gain_dark: float = 0.3,
    gain_bright: float = 3.0,
) -> list[np.ndarray]:
    """Alternates the masked region between two *multiplicative* luminance
    gains in linear light space, instead of inject_general_flash's flat
    color overwrite -- this preserves the underlying scene's texture, matching
    how a real light source's intensity change multiplies reflected light
    physically. Gain defaults validated empirically (see plan/spec) to
    reliably cross Detector's default thresholds on real footage."""
    rows, cols = _mask_slice(window)
    out = [frame.copy() for frame in frames]
    for i in range(window.start_frame, window.end_frame + 1):
        offset = i - window.start_frame
        gain = gain_bright if _is_pulse_frame(offset, window.period_frames) else gain_dark
        region = out[i][rows, cols, :]
        linear = _srgb_to_linear(region)
        out[i][rows, cols, :] = _linear_to_srgb(linear * gain)
    return out


def inject_red_flash_realistic(
    frames: list[np.ndarray],
    window: InjectionWindow,
    red_gains: tuple[float, float, float] = (20.0, 0.02, 0.02),
    baseline_gains: tuple[float, float, float] = (0.3, 1.0, 1.0),
) -> list[np.ndarray]:
    """Alternates the masked region between two per-channel gain triples in
    linear light space -- red_gains boosts R and suppresses G/B, baseline_gains
    suppresses R -- instead of inject_red_flash's flat saturated-color
    overwrite. Both states are deliberately engineered (neither is a pass-
    through of the original), because a pass-through baseline can itself
    exceed the saturation-ratio threshold on already-reddish content,
    breaking the XOR-based transition Detector relies on. gain magnitudes are
    large because the WCAG saturation-ratio test runs in gamma-compressed
    sRGB space, which compresses a 10x linear channel-ratio difference down
    to roughly 2.6x -- verified empirically (see plan/spec)."""
    rows, cols = _mask_slice(window)
    out = [frame.copy() for frame in frames]
    for i in range(window.start_frame, window.end_frame + 1):
        offset = i - window.start_frame
        gains = red_gains if _is_pulse_frame(offset, window.period_frames) else baseline_gains
        region = out[i][rows, cols, :]
        linear = _srgb_to_linear(region)
        modulated = linear * np.array(gains, dtype=np.float32)
        out[i][rows, cols, :] = _linear_to_srgb(modulated)
    return out
