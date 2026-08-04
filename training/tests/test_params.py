import pathlib

import numpy as np
import pytest

from detector.profiles import ThresholdProfile, load_profile
from training.params import ClipTooShortError, sample_synthesis_params

PROFILE = ThresholdProfile(
    name="test", max_flashes_per_second=3, max_area_ratio=0.10,
    general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
    red_saturation_ratio_threshold=0.80,
)

SHIPPED_PROFILES_DIR = pathlib.Path(__file__).parent.parent.parent / "configs" / "profiles"


def test_sample_synthesis_params_general_flash_targets_exceed_profile_thresholds():
    rng = np.random.default_rng(0)
    params = sample_synthesis_params(PROFILE, "general", clip_frame_count=90, frame_height=100, frame_width=100, fps=30.0, rng=rng)
    assert params.pattern == "general"
    assert params.dark_target < PROFILE.general_flash_dark_threshold
    assert params.bright_target - params.dark_target > PROFILE.general_flash_delta_threshold
    assert params.bright_target <= 1.0


def test_sample_synthesis_params_red_flash_targets_exceed_ratio_threshold():
    rng = np.random.default_rng(0)
    params = sample_synthesis_params(PROFILE, "red", clip_frame_count=90, frame_height=100, frame_width=100, fps=30.0, rng=rng)
    assert params.pattern == "red"
    r, g, b = params.red_rgb
    assert r / (r + g + b) > PROFILE.red_saturation_ratio_threshold
    br, bg, bb = params.baseline_rgb
    assert br / (br + bg + bb) < PROFILE.red_saturation_ratio_threshold


def test_sample_synthesis_params_mask_area_exceeds_profile_ratio():
    rng = np.random.default_rng(0)
    params = sample_synthesis_params(PROFILE, "general", clip_frame_count=90, frame_height=100, frame_width=100, fps=30.0, rng=rng)
    w = params.window
    mask_area_ratio = (w.mask_height * w.mask_width) / (100 * 100)
    assert mask_area_ratio > PROFILE.max_area_ratio
    assert 0 <= w.mask_top and w.mask_top + w.mask_height <= 100
    assert 0 <= w.mask_left and w.mask_left + w.mask_width <= 100


def test_sample_synthesis_params_period_yields_enough_flagged_frames_per_window():
    rng = np.random.default_rng(0)
    fps = 30.0
    params = sample_synthesis_params(PROFILE, "general", clip_frame_count=90, frame_height=100, frame_width=100, fps=fps, rng=rng)
    window_frames = round(fps)
    flagged_per_window = window_frames // params.window.period_frames
    assert flagged_per_window > PROFILE.max_flashes_per_second


def test_sample_synthesis_params_window_fits_inside_clip_and_covers_full_second():
    rng = np.random.default_rng(0)
    fps = 30.0
    clip_frame_count = 90
    params = sample_synthesis_params(PROFILE, "general", clip_frame_count=clip_frame_count, frame_height=100, frame_width=100, fps=fps, rng=rng)
    w = params.window
    assert 1 <= w.start_frame
    assert w.end_frame < clip_frame_count
    assert (w.end_frame - w.start_frame + 1) >= round(fps)
    assert w.ramp_frames == round(fps)
    assert (w.end_frame - w.start_frame + 1) - w.ramp_frames >= 2 * w.period_frames


def test_sample_synthesis_params_raises_when_clip_too_short():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_synthesis_params(PROFILE, "general", clip_frame_count=5, frame_height=100, frame_width=100, fps=30.0, rng=rng)


def test_clip_too_short_raises_the_specific_exception_type():
    # The CLI distinguishes "this clip is too short" from every other
    # ValueError (e.g. an unusable profile), so it needs its own type.
    rng = np.random.default_rng(0)
    with pytest.raises(ClipTooShortError):
        sample_synthesis_params(PROFILE, "general", clip_frame_count=5, frame_height=100, frame_width=100, fps=30.0, rng=rng)
    assert issubclass(ClipTooShortError, ValueError)


def test_sample_synthesis_params_rejects_profile_whose_area_target_is_unreachable():
    # _MAX_AREA_RATIO caps the mask at 90% of the frame, so a profile whose
    # own max_area_ratio is at/above that can never get an exceeding mask.
    # This must be a legible config error, not silent retry exhaustion.
    profile = ThresholdProfile(
        name="area-cliff", max_flashes_per_second=3, max_area_ratio=0.95,
        general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
        red_saturation_ratio_threshold=0.80,
    )
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="area-cliff"):
        sample_synthesis_params(profile, "general", clip_frame_count=90, frame_height=100, frame_width=100, fps=30.0, rng=rng)


def test_sample_synthesis_params_rejects_profile_whose_saturation_target_is_unreachable():
    profile = ThresholdProfile(
        name="sat-cliff", max_flashes_per_second=3, max_area_ratio=0.10,
        general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
        red_saturation_ratio_threshold=0.99,
    )
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="sat-cliff"):
        sample_synthesis_params(profile, "red", clip_frame_count=90, frame_height=100, frame_width=100, fps=30.0, rng=rng)


def test_sample_synthesis_params_rejects_profile_whose_delta_target_is_unreachable():
    # bright_target is clamped at 1.0, so a large delta threshold combined
    # with a high dark threshold leaves no room for the required delta.
    profile = ThresholdProfile(
        name="delta-cliff", max_flashes_per_second=3, max_area_ratio=0.10,
        general_flash_dark_threshold=1.00, general_flash_delta_threshold=0.60,
        red_saturation_ratio_threshold=0.80,
    )
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="delta-cliff"):
        sample_synthesis_params(profile, "general", clip_frame_count=90, frame_height=100, frame_width=100, fps=30.0, rng=rng)


@pytest.mark.parametrize("filename", ["kr.json", "jp.json", "itu.json", "ofcom.json", "w3c.json", "netflix.json"])
def test_sample_synthesis_params_exceeds_each_shipped_profile(filename):
    profile = load_profile(SHIPPED_PROFILES_DIR / filename)
    rng = np.random.default_rng(0)
    params = sample_synthesis_params(profile, "general", clip_frame_count=90, frame_height=100, frame_width=100, fps=30.0, rng=rng)
    mask_area_ratio = (params.window.mask_height * params.window.mask_width) / (100 * 100)
    assert mask_area_ratio > profile.max_area_ratio
    window_frames = round(30.0)
    flagged_per_window = window_frames // params.window.period_frames
    assert flagged_per_window > profile.max_flashes_per_second


def test_sample_synthesis_params_prefers_a_start_with_full_runway_when_clip_allows_it():
    # fps=24 -> window_frames=24, desired_runway=2*24=48. A 200-frame clip
    # gives latest_start = 200 - duration_frames, comfortably above 48, so
    # every draw should land at or beyond the runway target.
    rng = np.random.default_rng(0)
    for _ in range(20):
        params = sample_synthesis_params(
            PROFILE, "general", clip_frame_count=200, frame_height=100, frame_width=100, fps=24.0, rng=rng
        )
        assert params.window.start_frame >= 48


def test_sample_synthesis_params_squeezes_start_to_latest_when_clip_too_short_for_full_runway():
    # fps=24 -> window_frames=24; PROFILE's max_flashes_per_second=3 ->
    # period_frames = 24 // (2*4) = 3 -> duration_frames = 24 + 12 = 36.
    # clip_frame_count=40 -> latest_start = 40 - 36 = 4, well under the
    # desired_runway of 48 -- the clip cannot offer full runway, so
    # start_frame must be squeezed to exactly latest_start (the only value
    # that still leaves room for the full injection window).
    rng = np.random.default_rng(0)
    for _ in range(20):
        params = sample_synthesis_params(
            PROFILE, "general", clip_frame_count=40, frame_height=100, frame_width=100, fps=24.0, rng=rng
        )
        assert params.window.start_frame == 4
        assert params.window.end_frame == 4 + 36 - 1
