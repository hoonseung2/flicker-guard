"""Turns a per-frame FlickerScore sequence into merged risk segments with a
safety margin. A frame is risky only if BOTH flash frequency and flagged
area exceed the active profile's thresholds (per README.md section 2, most
regulatory rules combine a frequency condition with an area condition).

Both operands are windowed quantities over the same trailing one-second
window (final-review finding C2): `flagged_frame_count_last_second` against
`max_flagged_area_in_window`. ANDing the windowed frequency against the
current frame's instantaneous area — the pre-fix behaviour — meant that for
realistic duty-cycle strobing the two conditions rarely landed on the same
frame, fragmenting one continuous hazard into many short segments separated
by gaps reported as safe.

Note on units: `flagged_frame_count_last_second` counts flagged frames, ~2x
the visual flash rate (see `detector/scoring.py`), so comparing it against
the profile's regulatory `max_flashes_per_second` is intentionally about 2x
stricter than the regulation.
"""
from dataclasses import dataclass

from detector.profiles import ThresholdProfile
from detector.scoring import FlickerScore


@dataclass(eq=True)
class RiskSegment:
    start_frame: int
    end_frame: int


def _is_risky(score: FlickerScore, profile: ThresholdProfile) -> bool:
    return (
        score.flagged_frame_count_last_second > profile.max_flashes_per_second
        and score.max_flagged_area_in_window > profile.max_area_ratio
    )


def _merge_adjacent(segments: list[RiskSegment]) -> list[RiskSegment]:
    """Collapse overlapping or touching segments into one.

    Margin expansion can push two neighbouring runs into each other; emitting
    them separately would report a "safe" boundary inside a single hazard
    (final-review finding C2).
    """
    merged: list[RiskSegment] = []
    for segment in sorted(segments, key=lambda s: (s.start_frame, s.end_frame)):
        if merged and segment.start_frame <= merged[-1].end_frame + 1:
            merged[-1] = RiskSegment(
                start_frame=merged[-1].start_frame,
                end_frame=max(merged[-1].end_frame, segment.end_frame),
            )
        else:
            merged.append(segment)
    return merged


def scores_to_segments(
    scores: list[FlickerScore],
    profile: ThresholdProfile,
    margin_frames: int,
    total_frames: int,
) -> list[RiskSegment]:
    flagged = [_is_risky(s, profile) for s in scores]
    segments: list[RiskSegment] = []
    start = None
    for i, risky in enumerate(flagged):
        if risky and start is None:
            start = i
        elif not risky and start is not None:
            segments.append(RiskSegment(
                start_frame=max(0, start - margin_frames),
                end_frame=min(total_frames - 1, i - 1 + margin_frames),
            ))
            start = None
    if start is not None:
        segments.append(RiskSegment(
            start_frame=max(0, start - margin_frames),
            end_frame=min(total_frames - 1, len(flagged) - 1 + margin_frames),
        ))
    return _merge_adjacent(segments)
