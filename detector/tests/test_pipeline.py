import numpy as np
from detector.pipeline import run_detection
from detector.profiles import ThresholdProfile

PROFILE = ThresholdProfile(name="test", max_flashes_per_second=3, max_area_ratio=0.10)


def _solid_frame(value):
    return np.full((20, 20, 3), value, dtype=np.float32)


def test_run_detection_flags_strobing_sequence():
    dark, bright = _solid_frame(0.05), _solid_frame(0.95)
    frames = [dark, bright] * 8  # rapid strobing at fps
    scores, segments = run_detection(frames, fps=10, profile=PROFILE, margin_seconds=0.0)
    assert len(scores) == len(frames)
    assert len(segments) >= 1


def test_run_detection_leaves_static_sequence_unflagged():
    frames = [_solid_frame(0.5)] * 10
    scores, segments = run_detection(frames, fps=10, profile=PROFILE, margin_seconds=0.0)
    assert segments == []


def test_run_detection_applies_margin_in_frames():
    dark, bright = _solid_frame(0.05), _solid_frame(0.95)
    frames = [_solid_frame(0.5)] * 3 + [dark, bright] * 4 + [_solid_frame(0.5)] * 3
    scores, segments = run_detection(frames, fps=10, profile=PROFILE, margin_seconds=0.5)
    assert segments[0].start_frame < 3


def test_run_detection_ignores_pure_camera_pan():
    rng = np.random.default_rng(0)
    base = rng.random((40, 40, 3)).astype(np.float32)
    frames = [base]
    for i in range(1, 8):
        shifted = np.roll(base, shift=(i, i), axis=(0, 1))
        frames.append(shifted)
    scores, segments = run_detection(frames, fps=10, profile=PROFILE, margin_seconds=0.0)
    assert segments == []
