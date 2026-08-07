import numpy as np
import pytest
from detector.luminance import relative_luminance

def test_white_frame_has_luminance_one():
    frame = np.ones((4, 4, 3), dtype=np.float32)
    result = relative_luminance(frame)
    assert result.shape == (4, 4)
    assert np.allclose(result, 1.0, atol=1e-5)

def test_black_frame_has_luminance_zero():
    frame = np.zeros((4, 4, 3), dtype=np.float32)
    result = relative_luminance(frame)
    assert np.allclose(result, 0.0, atol=1e-5)

def test_mid_gray_matches_srgb_linearization():
    frame = np.full((2, 2, 3), 0.5, dtype=np.float32)
    result = relative_luminance(frame)
    # sRGB 0.5 linearizes to ((0.5+0.055)/1.055)**2.4 ≈ 0.21404
    assert np.allclose(result, 0.21404, atol=1e-4)

def test_pure_red_matches_bt709_red_coefficient():
    frame = np.zeros((2, 2, 3), dtype=np.float32)
    frame[..., 0] = 1.0  # pure red, R=1 G=0 B=0
    result = relative_luminance(frame)
    assert np.allclose(result, 0.2126, atol=1e-4)


def test_srgb_to_linear_round_trips_through_linear_to_srgb():
    from detector.luminance import linear_to_srgb, srgb_to_linear

    values = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    assert np.allclose(linear_to_srgb(srgb_to_linear(values)), values, atol=1e-6)


def test_linear_to_srgb_is_finite_for_negative_input():
    # Compression clamps its output before converting back, but a fractional
    # exponent on a negative base yields NaN, and a silent NaN frame is far
    # worse than a clamped one -- the conversion must not be the thing that
    # produces it.
    from detector.luminance import linear_to_srgb

    assert np.isfinite(linear_to_srgb(np.array([-0.5, 0.0, 0.5], dtype=np.float32))).all()


def test_srgb_linear_boundary_values_are_exact():
    from detector.luminance import linear_to_srgb, srgb_to_linear

    assert srgb_to_linear(np.array([0.0], dtype=np.float32))[0] == pytest.approx(0.0)
    assert srgb_to_linear(np.array([1.0], dtype=np.float32))[0] == pytest.approx(1.0)
    assert linear_to_srgb(np.array([0.0], dtype=np.float32))[0] == pytest.approx(0.0)
    assert linear_to_srgb(np.array([1.0], dtype=np.float32))[0] == pytest.approx(1.0)
