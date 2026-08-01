"""Global (pan-only) motion estimation/compensation to reduce false-positive
flash detections caused by camera panning, per the ITU-R BT.1702-04 reference
pipeline's motion-compensation stage."""
import cv2
import numpy as np


def estimate_global_shift(prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple[float, float]:
    """Estimate global shift (pan) between two grayscale frames using phase correlation.

    Args:
        prev_gray: Previous frame as (H, W) float32 grayscale array.
        curr_gray: Current frame as (H, W) float32 grayscale array.

    Returns:
        Tuple of (dx, dy) representing pixel shift in x and y directions.
    """
    (dx, dy), _ = cv2.phaseCorrelate(prev_gray.astype(np.float32), curr_gray.astype(np.float32))
    return dx, dy


def compensate_shift(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Apply affine transformation to compensate for global shift.

    Args:
        frame: Input frame as (H, W, C) or (H, W) float32 array.
        dx: Horizontal shift in pixels.
        dy: Vertical shift in pixels.

    Returns:
        Shifted frame with same shape as input, using linear interpolation
        and replicate border mode.
    """
    h, w = frame.shape[:2]
    matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
