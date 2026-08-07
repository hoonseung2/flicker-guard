import numpy as np
import pytest

from detector.luminance import relative_luminance
from detector.profiles import ThresholdProfile
from fallback.strength import motion_compensated_deltas, required_strength

PROFILE = ThresholdProfile(
    name="test", max_flashes_per_second=3, max_area_ratio=0.25,
    general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
    red_saturation_ratio_threshold=0.80,
)


def _texture(h=32, w=32, seed=0, scale=0.3):
    return (np.random.default_rng(seed).random((h, w, 3)).astype(np.float32) * scale)


def test_required_strength_matches_the_quantile_formula():
    # Half the pixels get a large brightness step, half stay put. With
    # max_area_ratio 0.25 the binding quantile is the 75th percentile of the
    # per-pixel delta, and the strength that brings it to the 0.10 threshold
    # is 1 - 0.10/q75. Computed here independently of the implementation.
    dark = _texture(seed=1)
    bright = dark.copy()
    bright[:, :16] = np.clip(bright[:, :16] + 0.6, 0.0, 1.0)

    delta = np.abs(relative_luminance(bright) - relative_luminance(dark))
    q75 = float(np.quantile(delta, 0.75))
    expected = 1.0 - 0.10 / q75

    assert required_strength([dark, bright], PROFILE) == pytest.approx(expected, abs=5e-3)


def test_required_strength_is_zero_for_an_already_safe_clip():
    # A static clip has no transitions at all, so no compression is owed.
    frame = _texture(seed=2)
    assert required_strength([frame, frame, frame], PROFILE) == 0.0


def test_required_strength_takes_the_worst_frame_pair():
    # One severe pair among mild ones must set the strength: if any pair
    # trips the area criterion the whole segment is flagged.
    base = _texture(seed=3)
    mild = np.clip(base + 0.02, 0.0, 1.0)
    severe = np.clip(base + 0.7, 0.0, 1.0)

    mild_only = required_strength([base, mild], PROFILE)
    with_severe = required_strength([base, mild, severe], PROFILE)
    assert with_severe > mild_only


def test_required_strength_never_exceeds_one():
    black = np.zeros((16, 16, 3), dtype=np.float32)
    white = np.ones((16, 16, 3), dtype=np.float32)
    assert 0.0 <= required_strength([black, white], PROFILE) <= 1.0


def test_required_strength_of_a_single_frame_is_zero():
    # No pair, no measurable transition -- must not raise on an empty
    # delta list.
    assert required_strength([_texture()], PROFILE) == 0.0


def test_motion_compensation_suppresses_deltas_from_a_pure_pan():
    # A translated copy of the same texture is not a flash. Without
    # compensation every edge pixel would register a large delta and the
    # estimate would demand needless compression on any panning shot.
    frame = _texture(h=64, w=64, seed=4, scale=0.9)
    shifted = np.roll(frame, shift=3, axis=1)

    compensated = motion_compensated_deltas([frame, shifted])[0]
    naive = np.abs(relative_luminance(shifted) - relative_luminance(frame))
    assert compensated.mean() < naive.mean() * 0.5


def test_motion_compensated_deltas_length_is_one_less_than_frames():
    frames = [_texture(seed=i) for i in range(4)]
    assert len(motion_compensated_deltas(frames)) == 3
