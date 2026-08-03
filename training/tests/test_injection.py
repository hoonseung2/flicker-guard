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
