import numpy as np
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
