import numpy as np

from detector.flash import red_flash_mask, transition_mask
from detector.luminance import relative_luminance
from training.injection import (
    inject_general_flash,
    inject_general_flash_realistic,
    inject_red_flash,
    inject_red_flash_realistic,
)
from training.params import InjectionWindow, LightShape


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


def _varied_frames(n, h, w):
    """Frames that differ from each other, so a copy-from-the-wrong-frame bug
    cannot hide behind a uniform-valued clip."""
    frames = []
    for i in range(n):
        frame = np.full((h, w, 3), 0.5, dtype=np.float32)
        frame[:, :, 0] = np.float32(0.01 * i)
        frame[i % h, :, 1] = np.float32(0.9)
        frames.append(frame)
    return frames


# start_frame > 0, a mask that is not the full frame, and period_frames > 1
# all interact: the offset->pulse mapping is relative to start_frame and the
# write is confined to the mask rectangle. Guards the byte-identity Global
# Constraint for that combination specifically.
def test_inject_general_flash_offset_window_submask_and_multiframe_period():
    frames = _varied_frames(20, 12, 12)
    originals = [f.copy() for f in frames]
    window = InjectionWindow(
        start_frame=4, end_frame=15, mask_top=2, mask_left=3,
        mask_height=5, mask_width=6, period_frames=3, ramp_frames=4,
    )
    out = inject_general_flash(frames, window, dark_target=0.2, bright_target=0.7)

    for i, original in enumerate(originals):
        assert np.array_equal(frames[i], original), f"input frame {i} mutated"

    for i in list(range(0, 4)) + list(range(16, 20)):
        assert np.array_equal(out[i], frames[i]), f"frame {i} outside window changed"

    rows, cols = slice(2, 7), slice(3, 9)
    for i in range(4, 16):
        restored = out[i].copy()
        restored[rows, cols, :] = frames[i][rows, cols, :]
        assert np.array_equal(restored, frames[i]), f"pixels outside mask changed in frame {i}"

    dark = float(out[4][2, 3, 0])
    bright = float(out[7][2, 3, 0])
    assert bright > dark
    # period_frames=3 -> 3 dark, 3 bright, 3 dark, 3 bright across offsets 0..11
    expected = [dark] * 3 + [bright] * 3 + [dark] * 3 + [bright] * 3
    for offset, value in enumerate(expected):
        region = out[4 + offset][rows, cols, :]
        assert region.min() == region.max() == np.float32(value), f"offset {offset}"


def test_inject_red_flash_offset_window_submask_and_multiframe_period():
    frames = _varied_frames(20, 12, 12)
    originals = [f.copy() for f in frames]
    window = InjectionWindow(
        start_frame=4, end_frame=15, mask_top=2, mask_left=3,
        mask_height=5, mask_width=6, period_frames=3, ramp_frames=4,
    )
    red, baseline = (0.9, 0.05, 0.05), (0.3, 0.3, 0.3)
    out = inject_red_flash(frames, window, red_rgb=red, baseline_rgb=baseline)

    for i, original in enumerate(originals):
        assert np.array_equal(frames[i], original), f"input frame {i} mutated"

    for i in list(range(0, 4)) + list(range(16, 20)):
        assert np.array_equal(out[i], frames[i]), f"frame {i} outside window changed"

    rows, cols = slice(2, 7), slice(3, 9)
    for i in range(4, 16):
        restored = out[i].copy()
        restored[rows, cols, :] = frames[i][rows, cols, :]
        assert np.array_equal(restored, frames[i]), f"pixels outside mask changed in frame {i}"

    expected = [baseline] * 3 + [red] * 3 + [baseline] * 3 + [red] * 3
    for offset, color in enumerate(expected):
        region = out[4 + offset][rows, cols, :]
        for channel, value in enumerate(color):
            assert region[:, :, channel].min() == region[:, :, channel].max() == np.float32(value), (
                f"offset {offset} channel {channel}"
            )


def test_inject_general_flash_realistic_leaves_frames_outside_window_untouched():
    frames = _solid_frames(10, 20, 20, 0.5)
    window = InjectionWindow(start_frame=3, end_frame=6, mask_top=0, mask_left=0, mask_height=20, mask_width=20, period_frames=1, ramp_frames=1)
    out = inject_general_flash_realistic(frames, window)
    assert np.array_equal(out[0], frames[0])
    assert np.array_equal(out[9], frames[9])


def test_inject_general_flash_realistic_leaves_pixels_outside_mask_untouched():
    frames = _solid_frames(6, 20, 20, 0.5)
    window = InjectionWindow(start_frame=1, end_frame=4, mask_top=5, mask_left=5, mask_height=5, mask_width=5, period_frames=1, ramp_frames=1)
    out = inject_general_flash_realistic(frames, window)
    assert np.array_equal(out[2][0:5, :, :], frames[2][0:5, :, :])
    assert frames[2][0, 0, 0] == 0.5  # input untouched


def test_inject_general_flash_realistic_preserves_texture():
    # Unlike inject_general_flash (flat overwrite), the two halves must stay
    # visibly different from each other after injection -- a flat overwrite
    # would collapse both to the same value.
    frame = np.zeros((10, 10, 3), dtype=np.float32)
    frame[0:5, :, :] = 0.2
    frame[5:10, :, :] = 0.6
    frames = [frame.copy() for _ in range(6)]
    window = InjectionWindow(start_frame=1, end_frame=4, mask_top=0, mask_left=0, mask_height=10, mask_width=10, period_frames=1, ramp_frames=1)
    out = inject_general_flash_realistic(frames, window, gain_dark=0.3, gain_bright=3.0)
    assert not np.allclose(out[2][0:5], out[2][5:10])


def test_inject_general_flash_realistic_transitions_are_flagged_by_detector():
    rng = np.random.default_rng(0)
    frame = rng.uniform(0.1, 0.9, size=(20, 20, 3)).astype(np.float32)
    frames = [frame.copy() for _ in range(8)]
    window = InjectionWindow(start_frame=1, end_frame=6, mask_top=0, mask_left=0, mask_height=20, mask_width=20, period_frames=1, ramp_frames=1)
    out = inject_general_flash_realistic(frames, window, gain_dark=0.3, gain_bright=3.0)
    lum2 = relative_luminance(out[2])
    lum3 = relative_luminance(out[3])
    mask = transition_mask(lum2, lum3, dark_threshold=0.80, delta_threshold=0.10)
    assert mask.mean() > 0.9  # measured 0.9875 on this exact fixture


def test_inject_red_flash_realistic_transitions_are_flagged_by_detector():
    rng = np.random.default_rng(0)
    frame = rng.uniform(0.1, 0.9, size=(10, 10, 3)).astype(np.float32)
    frames = [frame.copy() for _ in range(6)]
    window = InjectionWindow(start_frame=1, end_frame=4, mask_top=0, mask_left=0, mask_height=10, mask_width=10, period_frames=1, ramp_frames=1)
    out = inject_red_flash_realistic(frames, window)
    mask = red_flash_mask(out[2], out[3], saturation_ratio_threshold=0.80)
    assert mask.mean() > 0.9  # measured 0.95 on this exact fixture


def test_inject_general_flash_realistic_with_moving_light_hits_start_then_end_position():
    frames = _solid_frames(11, 30, 30, 0.5)
    light = LightShape(kind="circle", start_row=5, start_col=5, end_row=25, end_col=25, radius=2)
    window = InjectionWindow(
        start_frame=0, end_frame=10, mask_top=0, mask_left=0, mask_height=1, mask_width=1,
        period_frames=1, ramp_frames=1, lights=[light],
    )
    out = inject_general_flash_realistic(frames, window, gain_dark=0.3, gain_bright=3.0)
    # At start_frame the light sits near (5,5); its gain (dark, offset 0 is
    # not a pulse frame per _is_pulse_frame) must have changed that pixel...
    assert out[0][5, 5, 0] != frames[0][5, 5, 0]
    assert out[0][25, 25, 0] == frames[0][25, 25, 0]  # end position untouched yet
    # ...and at end_frame the light has moved to (25,25).
    assert out[10][25, 25, 0] != frames[10][25, 25, 0]
    assert out[10][5, 5, 0] == frames[10][5, 5, 0]  # start position no longer lit


def test_inject_general_flash_realistic_with_two_lights_unions_both_regions():
    frames = _solid_frames(4, 30, 30, 0.5)
    light_a = LightShape(kind="circle", start_row=5, start_col=5, end_row=5, end_col=5, radius=2)
    light_b = LightShape(kind="circle", start_row=25, start_col=25, end_row=25, end_col=25, radius=2)
    window = InjectionWindow(
        start_frame=0, end_frame=3, mask_top=0, mask_left=0, mask_height=1, mask_width=1,
        period_frames=1, ramp_frames=1, lights=[light_a, light_b],
    )
    out = inject_general_flash_realistic(frames, window, gain_dark=0.3, gain_bright=3.0)
    assert out[0][5, 5, 0] != frames[0][5, 5, 0]
    assert out[0][25, 25, 0] != frames[0][25, 25, 0]
    assert out[0][15, 15, 0] == frames[0][15, 15, 0]  # between the two lights: untouched


def test_inject_red_flash_realistic_with_moving_light_hits_start_then_end_position():
    frames = _solid_frames(11, 30, 30, 0.3)
    light = LightShape(kind="circle", start_row=5, start_col=5, end_row=25, end_col=25, radius=2)
    window = InjectionWindow(
        start_frame=0, end_frame=10, mask_top=0, mask_left=0, mask_height=1, mask_width=1,
        period_frames=1, ramp_frames=1, lights=[light],
    )
    out = inject_red_flash_realistic(frames, window)
    assert not np.array_equal(out[0][5, 5], frames[0][5, 5])
    assert np.array_equal(out[0][25, 25], frames[0][25, 25])
    assert not np.array_equal(out[10][25, 25], frames[10][25, 25])
    assert np.array_equal(out[10][5, 5], frames[10][5, 5])


def test_inject_general_flash_realistic_lights_leave_frames_outside_window_untouched():
    frames = _solid_frames(10, 20, 20, 0.5)
    light = LightShape(kind="rect", start_row=10, start_col=10, end_row=10, end_col=10, half_height=8, half_width=8)
    window = InjectionWindow(
        start_frame=3, end_frame=6, mask_top=0, mask_left=0, mask_height=1, mask_width=1,
        period_frames=1, ramp_frames=1, lights=[light],
    )
    out = inject_general_flash_realistic(frames, window)
    assert np.array_equal(out[0], frames[0])
    assert np.array_equal(out[9], frames[9])
