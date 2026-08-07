import numpy as np
import pytest

from detector.luminance import linear_to_srgb, relative_luminance, srgb_to_linear
from fallback.transfer import compress_contrast, trailing_reference


def _texture(h=16, w=16, seed=0):
    return (np.random.default_rng(seed).random((h, w, 3)).astype(np.float32) * 0.8 + 0.1)


def test_strength_zero_is_the_identity():
    frame = _texture()
    reference = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    assert np.allclose(compress_contrast(frame, reference, 0.0), frame, atol=1e-5)


def test_strength_one_makes_every_pixel_the_reference_colour():
    frame = _texture()
    reference = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    out = compress_contrast(frame, reference, 1.0)
    expected = linear_to_srgb(reference)
    assert np.allclose(out, expected[None, None, :], atol=1e-5)


def test_luminance_delta_scales_by_one_minus_strength():
    # This is the property the analytic strength estimate in strength.py
    # rests on: if it does not hold, the computed strength is meaningless.
    dark = _texture(seed=1) * 0.3
    bright = np.clip(dark + 0.4, 0.0, 1.0)
    reference = np.array([0.25, 0.25, 0.25], dtype=np.float32)

    before = np.abs(relative_luminance(bright) - relative_luminance(dark))
    for strength in (0.25, 0.5, 0.75):
        after = np.abs(
            relative_luminance(compress_contrast(bright, reference, strength))
            - relative_luminance(compress_contrast(dark, reference, strength))
        )
        assert np.allclose(after, before * (1.0 - strength), atol=1e-5)


def test_output_stays_in_range():
    frame = _texture()
    reference = np.array([0.9, 0.9, 0.9], dtype=np.float32)
    out = compress_contrast(frame, reference, 0.5)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_grey_input_with_grey_reference_stays_grey():
    # Compressing in sRGB space instead of linear light would pull the three
    # channels by different amounts and tint a neutral frame.
    frame = np.full((8, 8, 3), 0.7, dtype=np.float32)
    frame[:4] = 0.2
    reference = np.array([0.4, 0.4, 0.4], dtype=np.float32)
    out = compress_contrast(frame, reference, 0.6)
    assert np.allclose(out[..., 0], out[..., 1], atol=1e-6)
    assert np.allclose(out[..., 1], out[..., 2], atol=1e-6)


def test_strength_outside_the_unit_interval_is_rejected():
    frame = _texture()
    reference = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    with pytest.raises(ValueError):
        compress_contrast(frame, reference, 1.5)
    with pytest.raises(ValueError):
        compress_contrast(frame, reference, -0.1)


def test_trailing_reference_averages_the_window_ending_at_index():
    frames_linear = [np.full((4, 4, 3), value, dtype=np.float32) for value in (0.1, 0.2, 0.3, 0.9)]
    assert np.allclose(trailing_reference(frames_linear, 2, 3), [0.2, 0.2, 0.2], atol=1e-6)


def test_trailing_reference_clamps_at_the_clip_start():
    # Frame 0 has no history; averaging a window that runs off the front must
    # use what exists rather than index negatively and silently wrap around
    # to the end of the clip.
    frames_linear = [np.full((4, 4, 3), value, dtype=np.float32) for value in (0.1, 0.5, 0.9)]
    assert np.allclose(trailing_reference(frames_linear, 0, 5), [0.1, 0.1, 0.1], atol=1e-6)
