"""Per-pixel flash transition tests, per the WCAG 2.3.1 general flash and
red flash threshold definitions. MVP simplification: this module flags a
single directional transition; full-flash-cycle (opposing pair) counting
happens in scoring.py. Not a substitute for Harding FPA validation.

All thresholds arrive as parameters (final-review finding I3) — this module
holds no detection constants of its own. The defaults live in
`detector.profiles` so a single source of truth backs both the dataclass
fields and these signatures; `detector.pipeline` always passes the active
profile's values explicitly.
"""
import numpy as np

from detector.profiles import (
    DEFAULT_GENERAL_FLASH_DARK_THRESHOLD,
    DEFAULT_GENERAL_FLASH_DELTA_THRESHOLD,
    DEFAULT_RED_SATURATION_RATIO_THRESHOLD,
)


def transition_mask(
    prev_luminance: np.ndarray,
    curr_luminance: np.ndarray,
    dark_threshold: float = DEFAULT_GENERAL_FLASH_DARK_THRESHOLD,
    delta_threshold: float = DEFAULT_GENERAL_FLASH_DELTA_THRESHOLD,
) -> np.ndarray:
    darker = np.minimum(prev_luminance, curr_luminance)
    delta = np.abs(curr_luminance - prev_luminance)
    return (darker < dark_threshold) & (delta > delta_threshold)


def _is_saturated_red(
    frame_rgb: np.ndarray,
    saturation_ratio_threshold: float = DEFAULT_RED_SATURATION_RATIO_THRESHOLD,
) -> np.ndarray:
    """WCAG "saturated red": R / (R + G + B) >= threshold.

    Fixed by final-review finding I1 — the previous `R > 0.8 and G < 0.2 and
    B < 0.2` box test missed mid/dark saturated reds such as
    (0.60, 0.05, 0.05) (ratio 0.86). Pure black (R+G+B == 0) has no hue, so
    it is explicitly excluded instead of dividing by zero.
    """
    r, g, b = frame_rgb[..., 0], frame_rgb[..., 1], frame_rgb[..., 2]
    total = r + g + b
    ratio = np.zeros(r.shape, dtype=np.float64)
    np.divide(r, total, out=ratio, where=total > 0)
    return (total > 0) & (ratio >= saturation_ratio_threshold)


def red_flash_mask(
    prev_rgb: np.ndarray,
    curr_rgb: np.ndarray,
    saturation_ratio_threshold: float = DEFAULT_RED_SATURATION_RATIO_THRESHOLD,
) -> np.ndarray:
    return _is_saturated_red(prev_rgb, saturation_ratio_threshold) != _is_saturated_red(
        curr_rgb, saturation_ratio_threshold
    )
