"""Global (pan-only) motion estimation/compensation to reduce false-positive
flash detections caused by camera panning, per the ITU-R BT.1702-04 reference
pipeline's motion-compensation stage."""
import cv2
import numpy as np


# Below this correlation-peak height, phase correlation has not locked onto
# anything and the shift it reports is noise. Measured across 11 real clips
# at 480px: sound estimates score 0.35-0.84, garbage 0.03-0.23, and 0.30
# sits in the gap. The largest garbage shift observed was 1392.7 px on a
# 480-pixel frame, at response -0.000.
#
# The usual cause is not flicker -- injected flicker costs at most 0.99 px
# of error even at 80% frame area (spec section 8) -- but repetitive
# structure: a periodic pattern fits equally well one period over, so the
# peak splits.
_MIN_PHASE_CORRELATION_RESPONSE = 0.30


def estimate_global_shift_checked(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    min_response: float = _MIN_PHASE_CORRELATION_RESPONSE,
) -> tuple[float, float, bool]:
    """Estimate global shift and report, from the same correlation call,
    whether that estimate should be trusted.

    This is the one place the confidence guard is decided. `estimate_global_shift`
    delegates to it so a caller that only wants the (possibly zeroed) shift and
    a caller that also needs to know *why* it might be zero -- e.g. to avoid
    charging a rejected pan as flicker -- can never compute the guard
    differently: comparing the response against `min_response` in two places
    would let them drift out of sync if `min_response`'s default ever changed
    in one but not the other.

    Args:
        prev_gray: Previous frame as (H, W) float32 grayscale array.
        curr_gray: Current frame as (H, W) float32 grayscale array.
        min_response: Reject the estimate below this peak height. A
            non-positive value (e.g. 0.0) disables the guard entirely and
            treats the estimate as trusted regardless of response, including
            when the response itself is negative.

    Returns:
        Tuple of (dx, dy, trusted): the raw phase-correlation estimate and
        whether it cleared `min_response`. `dx, dy` are the raw estimate even
        when `trusted` is False -- callers that want the guarded (zeroed)
        shift should use `estimate_global_shift`.
    """
    (dx, dy), response = cv2.phaseCorrelate(
        prev_gray.astype(np.float32), curr_gray.astype(np.float32)
    )
    trusted = not (min_response > 0.0 and response < min_response)
    return dx, dy, trusted


def estimate_global_shift(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    min_response: float = _MIN_PHASE_CORRELATION_RESPONSE,
) -> tuple[float, float]:
    """Estimate global shift (pan) between two grayscale frames using phase correlation.

    Returns (0.0, 0.0) when the correlation peak is too weak to trust.
    Refusing to compensate is the pre-motion-stage behaviour and never
    compares unrelated pixels; applying an unfounded shift does.

    Args:
        prev_gray: Previous frame as (H, W) float32 grayscale array.
        curr_gray: Current frame as (H, W) float32 grayscale array.
        min_response: Reject the estimate below this peak height. A
            non-positive value (e.g. 0.0) disables the guard entirely and
            returns the raw estimate regardless of response, including when
            the response itself is negative.

    Returns:
        Tuple of (dx, dy) representing pixel shift in x and y directions.
    """
    dx, dy, trusted = estimate_global_shift_checked(prev_gray, curr_gray, min_response)
    if not trusted:
        return 0.0, 0.0
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
