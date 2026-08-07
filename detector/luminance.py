"""Relative luminance per WCAG/BT.709: sRGB -> linear -> weighted sum, plus
the sRGB transfer function in both directions.

The conversions are public because working in linear light is not a
detector-internal concern: any correction that scales or offsets pixel
values has to do it on linear values, or the sRGB curve turns a uniform
scale into a per-channel colour shift.
"""
import numpy as np

_SRGB_A = 0.055
_SRGB_LINEAR_CUTOFF = 0.0031308
_SRGB_ENCODED_CUTOFF = 0.04045
_R_COEFF, _G_COEFF, _B_COEFF = 0.2126, 0.7152, 0.0722


def srgb_to_linear(channel: np.ndarray) -> np.ndarray:
    low = channel <= _SRGB_ENCODED_CUTOFF
    # np.where evaluates both branches, so the power branch sees every
    # element including the ones the mask discards -- clamp its base at zero
    # so a negative input cannot produce NaN in the unselected half.
    safe = np.clip(channel, 0.0, None)
    return np.where(low, channel / 12.92, ((safe + _SRGB_A) / (1 + _SRGB_A)) ** 2.4)


def linear_to_srgb(channel: np.ndarray) -> np.ndarray:
    low = channel <= _SRGB_LINEAR_CUTOFF
    safe = np.clip(channel, 0.0, None)
    return np.where(low, channel * 12.92, (1 + _SRGB_A) * safe ** (1 / 2.4) - _SRGB_A)


def relative_luminance(frame_rgb: np.ndarray) -> np.ndarray:
    linear = srgb_to_linear(frame_rgb.astype(np.float32))
    r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]
    return _R_COEFF * r + _G_COEFF * g + _B_COEFF * b
