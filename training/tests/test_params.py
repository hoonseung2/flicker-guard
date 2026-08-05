import pathlib

import numpy as np
import pytest

from detector.profiles import ThresholdProfile, load_profile
from training.params import (
    _LIGHT_KINDS,
    _MAX_RED_BASELINE,
    _MAX_SATURATION_TARGET,
    _MIN_RED_TARGET,
    ClipTooShortError,
    LightShape,
    _achieved_area,
    _feasible_kinds,
    _light_area,
    _sample_one_light,
    sample_lights,
    sample_synthesis_params,
    sample_synthesis_params_realistic,
)

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
    # Luminance targets live in [0, 1], so the widest delta obtainable is
    # exactly 1.0 -- a profile demanding strictly more than that can never
    # be satisfied, however the targets are sampled.
    #
    # This used to use delta_threshold=0.60 with dark_threshold=1.00, which
    # the pre-randomization code could not satisfy only because it pinned
    # dark_target at half the dark threshold (0.5), leaving at most 0.5 of
    # headroom. Sampling dark_target down toward 0 finds the room that
    # profile always had, so it is no longer an example of an unreachable
    # delta.
    profile = ThresholdProfile(
        name="delta-cliff", max_flashes_per_second=3, max_area_ratio=0.10,
        general_flash_dark_threshold=1.00, general_flash_delta_threshold=1.00,
        red_saturation_ratio_threshold=0.80,
    )
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="delta-cliff"):
        sample_synthesis_params(profile, "general", clip_frame_count=90, frame_height=100, frame_width=100, fps=30.0, rng=rng)


def test_general_pattern_satisfiability_does_not_depend_on_the_draw():
    # dark_target is capped so a clearing bright_target always fits under
    # 1.0. Without that cap a low dark leaves room and a high one does not,
    # so the same profile would raise on some samples and not others --
    # random mid-run batch failures instead of a clean config error.
    profile = ThresholdProfile(
        name="tight-delta", max_flashes_per_second=3, max_area_ratio=0.10,
        general_flash_dark_threshold=1.00, general_flash_delta_threshold=0.60,
        red_saturation_ratio_threshold=0.80,
    )
    rng = np.random.default_rng(0)
    for _ in range(200):
        params = sample_synthesis_params(
            profile, "general", clip_frame_count=90, frame_height=100, frame_width=100, fps=30.0, rng=rng
        )
        assert params.bright_target - params.dark_target > profile.general_flash_delta_threshold


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


def test_sample_lights_returns_one_to_three_lights():
    rng = np.random.default_rng(0)
    counts = {len(sample_lights(PROFILE, 100, 100, rng)) for _ in range(30)}
    assert counts <= {1, 2, 3}
    assert counts  # at least one draw happened


def test_sample_lights_kinds_are_valid():
    rng = np.random.default_rng(0)
    for _ in range(10):
        for light in sample_lights(PROFILE, 100, 100, rng):
            assert light.kind in ("rect", "circle", "beam")


def test_sample_lights_combined_area_exceeds_profile_ratio():
    rng = np.random.default_rng(0)
    for _ in range(20):
        lights = sample_lights(PROFILE, 100, 100, rng)
        total_area = sum(_light_area(light) for light in lights)
        achieved_ratio = total_area / (100 * 100)
        assert achieved_ratio > PROFILE.max_area_ratio
        # Upper bound: per-light target areas are an equal share of at most
        # _MAX_LIGHTS_AREA_RATIO (0.60) of the frame, and each shape's
        # round()-derived half-extent should only overshoot its own target
        # by a few percent (not e.g. the ~2x a squared-term formula error
        # would produce -- see the beam half_thickness bug this test caught
        # during review). 1.0 is a generous, easy-to-justify ceiling: it is
        # far above any plausible few-percent-per-shape overshoot of 0.60,
        # yet would have caught the old beam formula (which pushed combined
        # area well past the frame's total pixel count on some profiles).
        assert achieved_ratio < 1.0


def test_sample_lights_positions_are_within_frame_bounds():
    # rng.uniform(0, frame_height) is half-open ([0, frame_height)), so
    # frame_height itself (e.g. row 80 for an 80-tall frame) is off-frame --
    # the upper bound must be strict, not <=.
    rng = np.random.default_rng(0)
    for _ in range(20):
        for light in sample_lights(PROFILE, 80, 60, rng):
            assert 0 <= light.start_row < 80 and 0 <= light.end_row < 80
            assert 0 <= light.start_col < 60 and 0 <= light.end_col < 60


def test_sample_lights_rejects_profile_whose_area_target_is_unreachable():
    profile = ThresholdProfile(
        name="area-cliff", max_flashes_per_second=3, max_area_ratio=0.95,
        general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
        red_saturation_ratio_threshold=0.80,
    )
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="area-cliff"):
        sample_lights(profile, 100, 100, rng)


def test_sample_synthesis_params_realistic_populates_lights():
    rng = np.random.default_rng(0)
    params = sample_synthesis_params_realistic(
        PROFILE, "general", clip_frame_count=200, frame_height=100, frame_width=100, fps=24.0, rng=rng
    )
    assert 1 <= len(params.window.lights) <= 3


def test_sample_synthesis_params_realistic_keeps_the_same_placement_and_timing_logic():
    # Same runway relaxation as plain sample_synthesis_params (Task 1) --
    # this wrapper must not re-derive start_frame/end_frame independently.
    rng = np.random.default_rng(0)
    for _ in range(20):
        params = sample_synthesis_params_realistic(
            PROFILE, "general", clip_frame_count=200, frame_height=100, frame_width=100, fps=24.0, rng=rng
        )
        assert params.window.start_frame >= 48


def test_sample_one_light_area_overshoot_is_bounded_for_every_kind():
    # Regression test for a beam half_thickness formula bug found in review:
    # dividing the target area by (2 * _BEAM_ASPECT) instead of
    # (4 * _BEAM_ASPECT) made half_thickness too large by sqrt(2), which
    # squares into ~2x area overshoot for beams -- while rect/circle only
    # overshoot their target by a few percent from round(). Exercised
    # directly against _sample_one_light (rather than through sample_lights)
    # with a large frame and targets well inside max_half, so the geometric
    # half_length clamp (frame_side // 2) can't mask the formula's own
    # overshoot behavior. Reviewer verified target=5000 on a 1000x1000 frame
    # achieved ratio~=2.05 pre-fix and ~=1.07 post-fix; this asserts the
    # fixed formula stays in the same few-percent range as rect/circle for
    # every kind and several target sizes.
    rng = np.random.default_rng(0)
    for target_area in (5000, 50000):
        for kind in ("rect", "circle", "beam"):
            light = _sample_one_light(kind, target_area, 1000, 1000, rng)
            overshoot = _light_area(light) / target_area
            assert 0.9 < overshoot < 1.3, f"{kind} at target={target_area}: overshoot={overshoot:.3f}"


def test_achieved_area_is_independent_of_the_random_positioning():
    # _achieved_area probes _sample_one_light with a throwaway generator, so
    # it is only a valid answer if a light's area depends on its size fields
    # alone and not on where the generator happened to place it.
    frame_height, frame_width = 480, 854
    target = 0.30 * frame_height * frame_width

    for kind in _LIGHT_KINDS:
        areas = {
            _light_area(_sample_one_light(kind, target, frame_height, frame_width, np.random.default_rng(seed)))
            for seed in range(10)
        }
        assert len(areas) == 1
        assert areas.pop() == _achieved_area(kind, target, frame_height, frame_width)


def test_feasible_kinds_keeps_every_shape_for_the_shipped_profiles_thresholds():
    # Shipped profiles sit at 0.10-0.25. Filtering exists for the extreme
    # end of the area range and must not quietly collapse shape variety in
    # the ordinary case -- a dataset of nothing but rectangles would undo
    # the earlier mask-diversification work.
    frame_height, frame_width = 480, 854
    frame_area = frame_height * frame_width

    for threshold in (0.10, 0.25):
        for target_ratio in (threshold + 0.10, 0.60):
            kinds = _feasible_kinds(
                target_ratio * frame_area, threshold * frame_area, frame_height, frame_width
            )
            assert set(kinds) == set(_LIGHT_KINDS)


def test_feasible_kinds_drops_a_shape_that_cannot_clear_a_high_threshold():
    # Measured on 480x854 at a 0.60 target: rect reaches 0.564 and beam
    # 0.519, but circle caps at 0.441 -- so against a 0.50 threshold the
    # circle is exactly the draw that used to raise.
    frame_height, frame_width = 480, 854
    frame_area = frame_height * frame_width

    kinds = _feasible_kinds(0.60 * frame_area, 0.50 * frame_area, frame_height, frame_width)

    assert "circle" not in kinds
    assert "rect" in kinds


def test_feasible_kinds_never_returns_empty():
    # sample_lights picks uniformly from this tuple -- an empty result would
    # be an IndexError instead of the clear _require_exceeds ValueError that
    # is supposed to report an unsatisfiable profile.
    frame_height, frame_width = 480, 854
    impossible_requirement = float(frame_height * frame_width) * 10
    assert _feasible_kinds(
        0.60 * frame_height * frame_width, impossible_requirement, frame_height, frame_width
    ) != ()


def test_sample_lights_does_not_raise_at_a_high_area_threshold():
    # Regression test: with uniform kind selection this raised ValueError on
    # roughly a third of draws at a 0.60 target (measured), and because it
    # is a plain ValueError rather than ClipTooShortError it aborted whole
    # batch runs rather than skipping one sample.
    profile = ThresholdProfile(
        name="wide-area", max_flashes_per_second=3, max_area_ratio=0.50,
        general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
        red_saturation_ratio_threshold=0.80,
    )
    rng = np.random.default_rng(0)
    for _ in range(200):
        assert len(sample_lights(profile, 480, 854, rng)) >= 1


def _area_ratio_of(lights, frame_height, frame_width):
    return sum(_light_area(light) for light in lights) / (frame_height * frame_width)


def test_sample_lights_produces_a_range_of_areas_not_one_fixed_value():
    # The whole point of this change: before it, every draw under a profile
    # produced essentially the same total area, so the model never saw a
    # localized flash and a near-full-screen one as different problems.
    rng = np.random.default_rng(0)
    frame_height, frame_width = 480, 854

    ratios = [
        _area_ratio_of(sample_lights(PROFILE, frame_height, frame_width, rng), frame_height, frame_width)
        for _ in range(200)
    ]

    assert max(ratios) - min(ratios) > 0.15


def test_sample_lights_area_always_exceeds_the_profile_threshold():
    # Randomizing area must never produce a sample that isn't actually
    # hazardous -- _require_exceeds guards this, and this test pins that the
    # guard still covers every draw.
    rng = np.random.default_rng(1)
    frame_height, frame_width = 480, 854

    for _ in range(200):
        lights = sample_lights(PROFILE, frame_height, frame_width, rng)
        assert _area_ratio_of(lights, frame_height, frame_width) > PROFILE.max_area_ratio


def test_sample_lights_handles_a_profile_whose_floor_meets_the_ceiling():
    # max_area_ratio 0.50 + _AREA_MARGIN 0.10 lands exactly on
    # _MAX_LIGHTS_AREA_RATIO 0.60, leaving an empty range -- must use the
    # floor rather than raising or sampling from an inverted interval.
    profile = ThresholdProfile(
        name="edge", max_flashes_per_second=3, max_area_ratio=0.50,
        general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
        red_saturation_ratio_threshold=0.80,
    )
    rng = np.random.default_rng(2)
    lights = sample_lights(profile, 480, 854, rng)
    assert _area_ratio_of(lights, 480, 854) > profile.max_area_ratio


_CONTRAST_PROFILE = ThresholdProfile(
    name="contrast", max_flashes_per_second=3, max_area_ratio=0.10,
    general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
    red_saturation_ratio_threshold=0.80,
)


def _sample_many(pattern, n=200, seed=0):
    rng = np.random.default_rng(seed)
    return [
        sample_synthesis_params(
            _CONTRAST_PROFILE, pattern, clip_frame_count=200,
            frame_height=480, frame_width=854, fps=24.0, rng=rng,
        )
        for _ in range(n)
    ]


def test_general_pattern_contrast_is_sampled_not_fixed():
    # Before this change every general-pattern example under a profile had
    # byte-identical dark/bright targets -- the minimum contrast that just
    # clears the threshold, never anything more severe.
    samples = _sample_many("general")
    assert len({s.dark_target for s in samples}) > 50
    assert len({s.bright_target for s in samples}) > 50


def test_general_pattern_contrast_always_clears_the_delta_threshold():
    # Randomizing must not let a sample fall below the profile's own
    # threshold -- it would be a "hazard" the Detector never flags.
    for s in _sample_many("general", seed=1):
        assert 0.0 <= s.dark_target <= _CONTRAST_PROFILE.general_flash_dark_threshold * 0.5
        assert s.bright_target <= 1.0
        assert s.bright_target - s.dark_target > _CONTRAST_PROFILE.general_flash_delta_threshold


def test_red_pattern_color_is_sampled_not_hardcoded():
    # r was a hardcoded 0.9 and baseline a hardcoded (0.3, 0.3, 0.3),
    # independent of profile -- every red example in the entire dataset was
    # the same color against the same gray.
    samples = _sample_many("red", seed=2)
    assert len({s.red_rgb[0] for s in samples}) > 50
    assert len({s.baseline_rgb[0] for s in samples}) > 50


def test_red_pattern_stays_within_its_bounds_and_clears_the_threshold():
    for s in _sample_many("red", seed=3):
        r, g, b = s.red_rgb
        assert _MIN_RED_TARGET <= r <= _MAX_SATURATION_TARGET
        base = s.baseline_rgb[0]
        assert 0.0 <= base <= _MAX_RED_BASELINE
        assert s.baseline_rgb == (base, base, base)
        # The saturation ratio the injected red actually achieves must
        # exceed the profile's threshold, or the sample isn't hazardous.
        assert r / (r + g + b) > _CONTRAST_PROFILE.red_saturation_ratio_threshold
