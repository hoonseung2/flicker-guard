import numpy as np

from prior.histogram import TargetHistogramSmoother, compute_illumination_histogram


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


def test_target_histogram_smoother_averages_only_clean_frames_in_window():
    smoother = TargetHistogramSmoother(window_frames=4)
    clean_hist = np.array([1.0, 0.0, 0.0, 0.0])
    flicker_hist = np.array([0.0, 0.0, 0.0, 1.0])

    smoother.update(clean_hist, is_risky_or_uncertain=False)
    smoother.update(clean_hist, is_risky_or_uncertain=False)
    target = smoother.update(flicker_hist, is_risky_or_uncertain=True)

    assert np.allclose(target, clean_hist)


def test_target_histogram_smoother_falls_back_to_last_clean_when_window_has_none():
    smoother = TargetHistogramSmoother(window_frames=2)
    clean_hist = np.array([1.0, 0.0])
    flicker_hist = np.array([0.0, 1.0])

    smoother.update(clean_hist, is_risky_or_uncertain=False)
    smoother.update(flicker_hist, is_risky_or_uncertain=True)
    target = smoother.update(flicker_hist, is_risky_or_uncertain=True)
    # window (size 2) now holds only the two flicker updates -> no clean frame in window
    assert np.allclose(target, clean_hist)


def test_target_histogram_smoother_returns_none_before_any_clean_frame_seen():
    smoother = TargetHistogramSmoother(window_frames=3)
    flicker_hist = np.array([0.0, 1.0])
    target = smoother.update(flicker_hist, is_risky_or_uncertain=True)
    assert target is None
