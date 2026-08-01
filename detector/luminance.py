"""Relative luminance per WCAG/BT.709: sRGB -> linear -> weighted sum."""
import numpy as np

_SRGB_A = 0.055
_R_COEFF, _G_COEFF, _B_COEFF = 0.2126, 0.7152, 0.0722


def _srgb_to_linear(channel: np.ndarray) -> np.ndarray:
    low = channel <= 0.04045
    return np.where(
        low,
        channel / 12.92,
        ((channel + _SRGB_A) / (1 + _SRGB_A)) ** 2.4,
    )


def relative_luminance(frame_rgb: np.ndarray) -> np.ndarray:
    linear = _srgb_to_linear(frame_rgb.astype(np.float32))
    r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]
    return _R_COEFF * r + _G_COEFF * g + _B_COEFF * b
