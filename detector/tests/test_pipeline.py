import numpy as np
from detector.flash import red_flash_mask, transition_mask
from detector.luminance import relative_luminance
from detector.motion import compensate_shift, estimate_global_shift
from detector.pipeline import run_detection, run_detection_with_masks
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


def test_run_detection_with_masks_matches_run_detection_scores_and_segments():
    dark, bright = _solid_frame(0.05), _solid_frame(0.95)
    frames = [dark, bright] * 8
    scores_a, segments_a = run_detection(frames, fps=10, profile=PROFILE, margin_seconds=0.0)
    scores_b, segments_b, masks = run_detection_with_masks(frames, fps=10, profile=PROFILE, margin_seconds=0.0)
    assert scores_a == scores_b
    assert segments_a == segments_b
    assert len(masks) == len(frames)
    assert masks[0].shape == (20, 20)


def test_run_detection_with_masks_frame_zero_is_all_true():
    frames = [_solid_frame(0.5)] * 5
    _, _, masks = run_detection_with_masks(frames, fps=10, profile=PROFILE, margin_seconds=0.0)
    assert masks[0].all()


def test_run_detection_with_masks_matches_manual_mask_computation():
    dark, bright = _solid_frame(0.05), _solid_frame(0.95)
    frames = [dark, bright, dark]
    _, _, masks = run_detection_with_masks(frames, fps=10, profile=PROFILE, margin_seconds=0.0)

    lum0 = relative_luminance(frames[0])
    lum1 = relative_luminance(frames[1])
    dx, dy = estimate_global_shift(lum0, lum1)
    aligned_prev_lum = compensate_shift(lum0, dx, dy)
    aligned_prev_rgb = compensate_shift(frames[0], dx, dy)
    expected_mask = transition_mask(
        aligned_prev_lum, lum1,
        dark_threshold=PROFILE.general_flash_dark_threshold,
        delta_threshold=PROFILE.general_flash_delta_threshold,
    ) | red_flash_mask(
        aligned_prev_rgb, frames[1],
        saturation_ratio_threshold=PROFILE.red_saturation_ratio_threshold,
    )
    assert (masks[1] == expected_mask).all()
