"""Per-pixel flash transition tests, per the WCAG 2.3.1 general flash and
red flash threshold definitions. MVP simplification: this module flags a
single directional transition; full-flash-cycle (opposing pair) counting
happens in scoring.py. Not a substitute for Harding FPA validation."""
import numpy as np

GENERAL_FLASH_DARK_THRESHOLD = 0.80
GENERAL_FLASH_DELTA_THRESHOLD = 0.10

_RED_R_MIN = 0.8
_RED_GB_MAX = 0.2


def transition_mask(prev_luminance: np.ndarray, curr_luminance: np.ndarray) -> np.ndarray:
    darker = np.minimum(prev_luminance, curr_luminance)
    delta = np.abs(curr_luminance - prev_luminance)
    return (darker < GENERAL_FLASH_DARK_THRESHOLD) & (delta > GENERAL_FLASH_DELTA_THRESHOLD)


def _is_saturated_red(frame_rgb: np.ndarray) -> np.ndarray:
    r, g, b = frame_rgb[..., 0], frame_rgb[..., 1], frame_rgb[..., 2]
    return (r > _RED_R_MIN) & (g < _RED_GB_MAX) & (b < _RED_GB_MAX)


def red_flash_mask(prev_rgb: np.ndarray, curr_rgb: np.ndarray) -> np.ndarray:
    return _is_saturated_red(prev_rgb) != _is_saturated_red(curr_rgb)
