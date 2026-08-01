from detector.scoring import FlickerScore
from detector.profiles import ThresholdProfile
from detector.segments import RiskSegment, scores_to_segments

PROFILE = ThresholdProfile(name="test", max_flashes_per_second=3, max_area_ratio=0.10)

def _score(i, flashes, area):
    return FlickerScore(frame_index=i, flash_count_last_second=flashes, flagged_area_ratio=area)

def test_single_risky_run_becomes_one_segment_with_margin():
    scores = [
        _score(0, 0, 0.0), _score(1, 0, 0.0),
        _score(2, 5, 0.20), _score(3, 5, 0.20), _score(4, 5, 0.20),
        _score(5, 0, 0.0), _score(6, 0, 0.0),
    ]
    segments = scores_to_segments(scores, PROFILE, margin_frames=1, total_frames=7)
    assert segments == [RiskSegment(start_frame=1, end_frame=5)]

def test_no_risky_frames_returns_empty_list():
    scores = [_score(i, 0, 0.0) for i in range(5)]
    assert scores_to_segments(scores, PROFILE, margin_frames=1, total_frames=5) == []

def test_risky_run_at_start_clamps_margin_to_zero():
    scores = [_score(0, 5, 0.20), _score(1, 5, 0.20), _score(2, 0, 0.0)]
    segments = scores_to_segments(scores, PROFILE, margin_frames=2, total_frames=3)
    assert segments == [RiskSegment(start_frame=0, end_frame=2)]

def test_risky_run_at_end_clamps_margin_to_total_frames():
    scores = [_score(0, 0, 0.0), _score(1, 5, 0.20), _score(2, 5, 0.20)]
    segments = scores_to_segments(scores, PROFILE, margin_frames=2, total_frames=3)
    assert segments == [RiskSegment(start_frame=0, end_frame=2)]

def test_both_flash_and_area_must_exceed_threshold():
    # flashes exceed but area doesn't -> not risky
    scores = [_score(0, 5, 0.05)]
    assert scores_to_segments(scores, PROFILE, margin_frames=0, total_frames=1) == []
