import numpy as np
import pytest

from detector.pipeline import run_detection
from detector.profiles import ThresholdProfile
from fallback.apply import FallbackReport, fade_weights, mitigate_with_fallback

PROFILE = ThresholdProfile(
    name="test", max_flashes_per_second=3, max_area_ratio=0.25,
    general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
    red_saturation_ratio_threshold=0.80,
)


def test_fade_starts_and_ends_near_zero_and_is_full_in_the_middle():
    weights = fade_weights(length=40, margin_frames=10)
    assert weights[0] < 0.1
    assert weights[-1] < 0.1
    assert weights[20] == pytest.approx(1.0)


def test_fade_is_monotonic_on_the_way_in_and_out():
    weights = fade_weights(length=40, margin_frames=10)
    assert np.all(np.diff(weights[:10]) > 0)
    assert np.all(np.diff(weights[-10:]) < 0)


def test_fade_never_exceeds_one_or_drops_below_zero():
    for length, margin in ((40, 10), (5, 10), (1, 10), (2, 1)):
        weights = fade_weights(length, margin)
        assert weights.min() >= 0.0 and weights.max() <= 1.0
        assert len(weights) == length


def test_fade_halves_itself_when_the_segment_is_shorter_than_two_margins():
    # A 6-frame segment with a 10-frame margin would otherwise write two
    # overlapping 10-frame ramps into a 6-slot array, and the second would
    # overwrite the first -- producing a step exactly where the fade exists
    # to prevent one.
    weights = fade_weights(length=6, margin_frames=10)
    assert np.all(np.diff(weights[:3]) > 0)
    assert np.all(np.diff(weights[3:]) < 0)
