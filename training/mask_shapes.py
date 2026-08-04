"""Renders boolean pixel masks for the shapes DatasetSynth's realistic
injection path can move around a frame (rectangle, circle, beam), plus the
analytic pixel area each shape occupies.

Kept independent of InjectionWindow/LightShape (training.params) and of
color logic (training.injection) so shape geometry has exactly one place to
live and one thing to test -- neither of those modules' tests need to
reason about rasterization, and this module's tests never need a
ThresholdProfile or an RNG.
"""
import numpy as np


def rect_area(half_height: int, half_width: int) -> int:
    return (2 * half_height + 1) * (2 * half_width + 1)


def circle_area(radius: int) -> int:
    return round(np.pi * radius ** 2)


def beam_area(half_length: int, half_thickness: int) -> int:
    return (2 * half_length + 1) * (2 * half_thickness + 1)


def render_rect_mask(
    frame_height: int,
    frame_width: int,
    center_row: float,
    center_col: float,
    half_height: int,
    half_width: int,
) -> np.ndarray:
    rows = np.arange(frame_height)[:, None].astype(np.float64)
    cols = np.arange(frame_width)[None, :].astype(np.float64)
    return (np.abs(rows - center_row) <= half_height) & (np.abs(cols - center_col) <= half_width)


def render_circle_mask(
    frame_height: int,
    frame_width: int,
    center_row: float,
    center_col: float,
    radius: int,
) -> np.ndarray:
    rows = np.arange(frame_height)[:, None].astype(np.float64)
    cols = np.arange(frame_width)[None, :].astype(np.float64)
    return (rows - center_row) ** 2 + (cols - center_col) ** 2 <= radius ** 2


def render_beam_mask(
    frame_height: int,
    frame_width: int,
    center_row: float,
    center_col: float,
    half_length: int,
    half_thickness: int,
    angle_degrees: float,
) -> np.ndarray:
    rows = np.arange(frame_height)[:, None].astype(np.float64) - center_row
    cols = np.arange(frame_width)[None, :].astype(np.float64) - center_col
    theta = np.deg2rad(angle_degrees)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Rotate frame coordinates into the beam's own axes so the beam is an
    # axis-aligned rectangle (length along local_long, thickness along
    # local_short) in its own frame -- same trick as rotating a bounding box.
    local_long = cols * cos_t + rows * sin_t
    local_short = -cols * sin_t + rows * cos_t
    return (np.abs(local_long) <= half_length) & (np.abs(local_short) <= half_thickness)
