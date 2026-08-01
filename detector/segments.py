"""Turns a per-frame FlickerScore sequence into merged risk segments with a
safety margin. A frame is risky only if BOTH flash frequency and flagged
area exceed the active profile's thresholds (per README.md section 2, most
regulatory rules combine a frequency condition with an area condition)."""
from dataclasses import dataclass

from detector.profiles import ThresholdProfile
from detector.scoring import FlickerScore


@dataclass(eq=True)
class RiskSegment:
    start_frame: int
    end_frame: int


def _is_risky(score: FlickerScore, profile: ThresholdProfile) -> bool:
    return (
        score.flash_count_last_second > profile.max_flashes_per_second
        and score.flagged_area_ratio > profile.max_area_ratio
    )


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
    return segments
