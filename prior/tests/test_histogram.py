import numpy as np

from prior.histogram import compute_illumination_histogram


def test_compute_illumination_histogram_sums_to_one():
    rng = np.random.default_rng(0)
    luminance = rng.random((10, 10)).astype(np.float32)
    hist = compute_illumination_histogram(luminance, n_bins=64)
    assert hist.shape == (64,)
    assert np.isclose(hist.sum(), 1.0)


def test_compute_illumination_histogram_all_zero_frame_fills_first_bin():
    luminance = np.zeros((5, 5), dtype=np.float32)
    hist = compute_illumination_histogram(luminance, n_bins=64)
    assert hist[0] == 1.0
    assert hist[1:].sum() == 0.0


def test_compute_illumination_histogram_half_and_half():
    luminance = np.zeros((4, 4), dtype=np.float32)
    luminance[:2, :] = 1.0  # half at 0.0 (bin 0), half at 1.0 (last bin)
    hist = compute_illumination_histogram(luminance, n_bins=4)
    assert np.isclose(hist[0], 0.5)
    assert np.isclose(hist[-1], 0.5)
