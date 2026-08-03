import numpy as np

from detector.flash import red_flash_mask, transition_mask
from detector.luminance import relative_luminance
from training.injection import inject_general_flash, inject_red_flash
from training.params import InjectionWindow


def _solid_frames(n, h, w, value):
    return [np.full((h, w, 3), value, dtype=np.float32) for _ in range(n)]


def test_inject_general_flash_leaves_frames_outside_window_untouched():
    frames = _solid_frames(10, 20, 20, 0.5)
    window = InjectionWindow(start_frame=3, end_frame=6, mask_top=0, mask_left=0, mask_height=20, mask_width=20, period_frames=1, ramp_frames=1)
    out = inject_general_flash(frames, window, dark_target=0.1, bright_target=0.5)
    assert np.array_equal(out[0], frames[0])
    assert np.array_equal(out[2], frames[2])
    assert np.array_equal(out[7], frames[7])
    assert np.array_equal(out[9], frames[9])


def test_inject_general_flash_leaves_pixels_outside_mask_untouched():
    frames = _solid_frames(6, 20, 20, 0.5)
    window = InjectionWindow(start_frame=1, end_frame=4, mask_top=5, mask_left=5, mask_height=5, mask_width=5, period_frames=1, ramp_frames=1)
    out = inject_general_flash(frames, window, dark_target=0.1, bright_target=0.5)
    assert np.array_equal(out[2][0:5, :, :], frames[2][0:5, :, :])
    assert frames[2][0, 0, 0] == 0.5  # input untouched


def test_inject_general_flash_transitions_are_flagged_by_detector():
    frames = _solid_frames(8, 20, 20, 0.5)
    window = InjectionWindow(start_frame=1, end_frame=6, mask_top=0, mask_left=0, mask_height=20, mask_width=20, period_frames=1, ramp_frames=1)
    out = inject_general_flash(frames, window, dark_target=0.4, bright_target=0.6)
    lum2 = relative_luminance(out[2])
    lum3 = relative_luminance(out[3])
    mask = transition_mask(lum2, lum3, dark_threshold=0.80, delta_threshold=0.10)
    assert mask.all()


def test_inject_red_flash_transitions_are_flagged_by_detector():
    frames = _solid_frames(6, 10, 10, 0.3)
    window = InjectionWindow(start_frame=1, end_frame=4, mask_top=0, mask_left=0, mask_height=10, mask_width=10, period_frames=1, ramp_frames=1)
    out = inject_red_flash(frames, window, red_rgb=(0.9, 0.05, 0.05), baseline_rgb=(0.3, 0.3, 0.3))
    mask = red_flash_mask(out[2], out[3], saturation_ratio_threshold=0.80)
    assert mask.all()
