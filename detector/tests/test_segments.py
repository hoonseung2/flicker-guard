from detector.scoring import FlickerScore
from detector.profiles import ThresholdProfile
from detector.segments import RiskSegment, scores_to_segments

PROFILE = ThresholdProfile(name="test", max_flashes_per_second=3, max_area_ratio=0.10)

def _score(i, flashes, area, window_area=None):
    return FlickerScore(
        frame_index=i,
        flagged_frame_count_last_second=flashes,
        flagged_area_ratio=area,
        max_flagged_area_in_window=area if window_area is None else window_area,
    )

def assert_valid_segments(segments, total_frames):
    """Shared invariant: sorted, non-overlapping, non-touching, in bounds."""
    previous_end = None
    for segment in segments:
        assert 0 <= segment.start_frame <= segment.end_frame <= total_frames - 1, segment
        if previous_end is not None:
            assert segment.start_frame > previous_end + 1, (segments, "overlapping/adjacent")
        previous_end = segment.end_frame

def test_single_risky_run_becomes_one_segment_with_margin():
    scores = [
        _score(0, 0, 0.0), _score(1, 0, 0.0),
        _score(2, 5, 0.20), _score(3, 5, 0.20), _score(4, 5, 0.20),
        _score(5, 0, 0.0), _score(6, 0, 0.0),
    ]
    segments = scores_to_segments(scores, PROFILE, margin_frames=1, total_frames=7)
    assert segments == [RiskSegment(start_frame=1, end_frame=5)]
    assert_valid_segments(segments, total_frames=7)

def test_no_risky_frames_returns_empty_list():
    scores = [_score(i, 0, 0.0) for i in range(5)]
    assert scores_to_segments(scores, PROFILE, margin_frames=1, total_frames=5) == []

def test_risky_run_at_start_clamps_margin_to_zero():
    scores = [_score(0, 5, 0.20), _score(1, 5, 0.20), _score(2, 0, 0.0)]
    segments = scores_to_segments(scores, PROFILE, margin_frames=2, total_frames=3)
    assert segments == [RiskSegment(start_frame=0, end_frame=2)]
    assert_valid_segments(segments, total_frames=3)

def test_risky_run_at_end_clamps_margin_to_total_frames():
    scores = [_score(0, 0, 0.0), _score(1, 5, 0.20), _score(2, 5, 0.20)]
    segments = scores_to_segments(scores, PROFILE, margin_frames=2, total_frames=3)
    assert segments == [RiskSegment(start_frame=0, end_frame=2)]
    assert_valid_segments(segments, total_frames=3)

def test_both_flash_and_area_must_exceed_threshold():
    # flashes exceed but area doesn't -> not risky
    scores = [_score(0, 5, 0.05)]
    assert scores_to_segments(scores, PROFILE, margin_frames=0, total_frames=1) == []

def test_risk_uses_windowed_area_not_instantaneous_area():
    # C2: a duty-cycle strobe's off-frames carry no instantaneous flagged area,
    # but the window still holds a big-area flash, so they stay risky.
    scores = [
        _score(0, 0, 0.0),
        _score(1, 5, 0.90, window_area=0.90),
        _score(2, 5, 0.00, window_area=0.90),
        _score(3, 5, 0.00, window_area=0.90),
        _score(4, 0, 0.0),
    ]
    segments = scores_to_segments(scores, PROFILE, margin_frames=0, total_frames=5)
    assert segments == [RiskSegment(start_frame=1, end_frame=3)]
    assert_valid_segments(segments, total_frames=5)

def test_margin_expansion_merges_neighbouring_runs():
    # C2: two runs one frame apart expand into each other; emitting both would
    # report a "safe" boundary inside a single hazard.
    scores = [
        _score(0, 0, 0.0),
        _score(1, 5, 0.20), _score(2, 5, 0.20),
        _score(3, 0, 0.0), _score(4, 0, 0.0), _score(5, 0, 0.0),
        _score(6, 5, 0.20), _score(7, 5, 0.20),
        _score(8, 0, 0.0),
    ]
    segments = scores_to_segments(scores, PROFILE, margin_frames=2, total_frames=9)
    assert segments == [RiskSegment(start_frame=0, end_frame=8)]
    assert_valid_segments(segments, total_frames=9)

def test_touching_segments_are_merged():
    # margin 1 leaves segment A as [0, 1] and segment B as [2, 4]: they touch
    # with no uncovered frame between them, so they are one hazard.
    scores = [
        _score(0, 5, 0.20),
        _score(1, 0, 0.0), _score(2, 0, 0.0),
        _score(3, 5, 0.20),
        _score(4, 0, 0.0),
    ]
    segments = scores_to_segments(scores, PROFILE, margin_frames=1, total_frames=5)
    assert segments == [RiskSegment(start_frame=0, end_frame=4)]
    assert_valid_segments(segments, total_frames=5)

def test_distant_runs_stay_separate():
    scores = (
        [_score(0, 5, 0.20)]
        + [_score(i, 0, 0.0) for i in range(1, 10)]
        + [_score(10, 5, 0.20), _score(11, 0, 0.0)]
    )
    segments = scores_to_segments(scores, PROFILE, margin_frames=1, total_frames=12)
    assert segments == [
        RiskSegment(start_frame=0, end_frame=1),
        RiskSegment(start_frame=9, end_frame=11),
    ]
    assert_valid_segments(segments, total_frames=12)

def test_entire_clip_risky_produces_one_full_span_segment():
    scores = [_score(i, 5, 0.20) for i in range(4)]
    segments = scores_to_segments(scores, PROFILE, margin_frames=3, total_frames=4)
    assert segments == [RiskSegment(start_frame=0, end_frame=3)]
    assert_valid_segments(segments, total_frames=4)
