import numpy as np

from detector.luminance import relative_luminance
from detector.profiles import ThresholdProfile
from prior.compute import compute_prior
from prior.histogram import compute_illumination_histogram
from training.injection import inject_general_flash
from training.params import InjectionWindow

PROFILE = ThresholdProfile(
    name="test", max_flashes_per_second=3, max_area_ratio=0.10,
    general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
    red_saturation_ratio_threshold=0.80,
)


def _clean_clip(n, h=20, w=20, value=0.5):
    return [np.full((h, w, 3), value, dtype=np.float32) for _ in range(n)]


def test_compute_prior_marks_injected_window_pixels_in_mask():
    clean = _clean_clip(n=30)
    window = InjectionWindow(
        start_frame=5, end_frame=20, mask_top=0, mask_left=0,
        mask_height=20, mask_width=20, period_frames=1, ramp_frames=10,
    )
    degraded = inject_general_flash(clean, window, dark_target=0.2, bright_target=0.8)

    results = compute_prior(degraded, fps=10.0, profile=PROFILE)

    assert len(results) == 30
    assert results[6].mask.all()


def test_compute_prior_target_histogram_returns_to_baseline_well_after_injection():
    clean = _clean_clip(n=80, value=0.5)
    window = InjectionWindow(
        start_frame=10, end_frame=20, mask_top=0, mask_left=0,
        mask_height=20, mask_width=20, period_frames=1, ramp_frames=10,
    )
    degraded = inject_general_flash(clean, window, dark_target=0.2, bright_target=0.8)

    results = compute_prior(degraded, fps=10.0, profile=PROFILE)

    baseline_histogram = compute_illumination_histogram(relative_luminance(clean[0]), n_bins=64)
    late_target = results[75].target_histogram
    assert late_target is not None
    assert np.allclose(late_target, baseline_histogram, atol=1e-6)


def test_compute_prior_target_never_picks_up_the_bright_pulse_during_injected_flicker():
    clean = _clean_clip(n=80, value=0.5)
    window = InjectionWindow(
        start_frame=40, end_frame=50, mask_top=0, mask_left=0,
        mask_height=20, mask_width=20, period_frames=1, ramp_frames=5,
    )
    degraded = inject_general_flash(clean, window, dark_target=0.2, bright_target=0.8)
    results = compute_prior(degraded, fps=10.0, profile=PROFILE)

    bright_pulse_histogram = compute_illumination_histogram(relative_luminance(degraded[41]), n_bins=64)
    # Frame 40 (the pattern's first, dark frame) transitions dark-baseline ->
    # slightly-darker, which the detector's transition_mask correctly does
    # not flag as a flash -- so its own near-baseline content legitimately
    # becomes the held reference for the rest of the fully-masked
    # dark/bright alternation that follows. Frame 45 sits deep in that
    # stretch; the one thing that must never happen is the DANGEROUS bright
    # pulse leaking into the target the network is asked to correct toward.
    assert not np.allclose(results[45].target_histogram, bright_pulse_histogram, atol=1e-6)


def test_compute_prior_returns_frame_prior_per_frame_with_matching_indices():
    clean = _clean_clip(n=10)
    results = compute_prior(clean, fps=10.0, profile=PROFILE)
    assert [r.frame_index for r in results] == list(range(10))
    assert results[0].target_histogram is None  # frame 0 always uncertain, no clean frame seen yet


def test_compute_prior_target_is_a_trailing_average_not_the_latest_frame_alone():
    # Regression test for the hybrid design: target histograms are derived
    # from each frame's own non-masked pixels (so a continuously-risky clip
    # never starves), but smoothed over a trailing window (like the old
    # temporal approach) so the target doesn't jitter frame-to-frame with
    # whatever the current frame happens to look like.
    frames = []
    for i in range(12):
        value = 0.5 if i % 2 == 0 else 0.55
        frames.append(np.full((20, 20, 3), value, dtype=np.float32))

    results = compute_prior(frames, fps=4.0, profile=PROFILE)  # window_frames = round(4.0) = 4

    hist_a = compute_illumination_histogram(relative_luminance(frames[10]), n_bins=64)  # value=0.5
    hist_b = compute_illumination_histogram(relative_luminance(frames[11]), n_bins=64)  # value=0.55
    expected = np.mean([hist_a, hist_b, hist_a, hist_b], axis=0)  # last 4 contributing frames: 8,9,10,11

    assert np.allclose(results[11].target_histogram, expected, atol=1e-6)
    # Not just the latest frame's own histogram -- that would mean no
    # smoothing is happening at all.
    assert not np.allclose(results[11].target_histogram, hist_b, atol=1e-6)


def test_compute_prior_accepts_a_one_shot_generator():
    # detector.cli.read_video_frames returns a generator; compute_prior consumes
    # frames twice, so it must materialise them rather than silently returning [].
    frames = (frame for frame in _clean_clip(n=12))
    results = compute_prior(frames, fps=10.0, profile=PROFILE)
    assert len(results) == 12
    assert [r.frame_index for r in results] == list(range(12))
