import numpy as np
from detector.pipeline import run_detection
from detector.profiles import ThresholdProfile

from detector.tests.test_segments import assert_valid_segments

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


def test_run_detection_reports_duty_cycle_strobe_as_one_contiguous_segment():
    # C2: a 6 Hz full-screen strobe at 30 fps (2 bright frames, 3 dark frames
    # per 5-frame period) is one continuous hazard. ANDing the windowed flash
    # count against the *current frame's* area fragmented it into ~16 segments
    # with "safe" gaps mid-hazard; both operands are now windowed.
    dark, bright = _solid_frame(0.05), _solid_frame(0.95)
    frames = ([bright] * 2 + [dark] * 3) * 18  # 90 frames = 3 s at 30 fps
    scores, segments = run_detection(frames, fps=30, profile=PROFILE, margin_seconds=0.0)
    assert len(segments) == 1, segments
    assert_valid_segments(segments, total_frames=len(frames))
    # and it must cover essentially the whole strobing stretch
    covered = segments[0].end_frame - segments[0].start_frame + 1
    assert covered >= len(frames) - 5, segments


def test_run_detection_treats_first_frame_as_uncertain_not_safe():
    # I4: frame 0 has no predecessor, so its transition is unknown. It must not
    # be scored as a definitely-safe all-zeros mask.
    frames = [_solid_frame(0.5)] * 12
    scores, _ = run_detection(frames, fps=10, profile=PROFILE, margin_seconds=0.0)
    assert scores[0].uncertain is True
    assert scores[0].flagged_area_ratio == 1.0
    assert scores[0].flagged_frame_count_last_second == 10  # whole window unknown
    # once frame 0 has slid out of the trailing window, nothing is assumed
    assert scores[-1].uncertain is False
    assert scores[-1].flagged_frame_count_last_second == 0


def test_run_detection_consumes_a_streaming_iterable():
    # I6: run_detection must never require the whole video in RAM. Feeding it a
    # one-shot generator (no len(), single pass) must work.
    dark, bright = _solid_frame(0.05), _solid_frame(0.95)
    materialized = [dark, bright] * 8

    def _stream():
        yield from materialized

    stream_scores, stream_segments = run_detection(
        _stream(), fps=10, profile=PROFILE, margin_seconds=0.0
    )
    list_scores, list_segments = run_detection(
        materialized, fps=10, profile=PROFILE, margin_seconds=0.0
    )
    assert len(stream_scores) == len(materialized)
    assert stream_scores == list_scores
    assert stream_segments == list_segments


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
