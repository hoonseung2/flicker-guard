"""Aggregates per-pixel transition masks into windowed flash-frequency and
flagged-area scores over a trailing one-second window.

Units (final-review finding I2): `flagged_frame_count_last_second` counts
*flagged frames* in the window, not visual flashes. One visual flash produces
two flagged frames (onset + offset), so for symmetric strobing this number
runs at roughly **2x the true flash rate**; for a flash held across several
frames it runs higher still. Comparing it against a profile's
`max_flashes_per_second` (a regulatory flashes-per-second limit) is therefore
deliberately conservative — it over-flags by roughly 2x and never
under-flags. It was named `flash_count_last_second` until finding I2; the
counting semantics themselves date from fix commit 612cd33, which replaced
onset-edge counting (which collapsed a sustained strobe to 1) with
flagged-frame counting.

Temporal scope (final-review finding C2): both risk operands are windowed.
`max_flagged_area_in_window` is the largest flagged area seen anywhere in the
trailing window, so `segments._is_risky` no longer ANDs a windowed frequency
against a single-frame area — a duty-cycle strobe whose bright frames and
flagged frames rarely coincide used to fragment into many short segments with
"safe" gaps mid-hazard.

Uncertain frames (final-review finding I4, README section 7 "프레임 부족으로
측정이 불확실한 구간은 보수적으로 위험 취급"): a frame with no measurable
predecessor (frame 0) and the not-yet-elapsed slots of the very first window
are unknown, not safe. They are counted as flagged for the frequency
statistic, which can only push the verdict toward "risky". Their *area*,
however, is recorded as unmeasured and excluded from
`max_flagged_area_in_window`: fabricating a full-screen area for them would,
combined with the known motion-compensation border residual, mark the first
second of every panning clip as hazardous and defeat the pan
false-positive reduction that the motion stage exists for. Every score whose
window still overlaps an unmeasured slot carries `uncertain=True` so
downstream consumers (BufferManager / Fallback, README section 7) can apply
their own conservative policy.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class FlickerScore:
    frame_index: int
    flagged_frame_count_last_second: int
    flagged_area_ratio: float
    max_flagged_area_in_window: float = 0.0
    uncertain: bool = False


def flagged_area_ratio(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask)) / mask.size


class WindowedFlashCounter:
    def __init__(self, fps: float):
        self.fps = fps
        self.window_frames = max(1, round(fps))
        # (flagged, area) per frame in the trailing window; area is None when
        # the frame carries no usable measurement. The window is primed with
        # unmeasured-but-flagged slots so the frequency statistic is not
        # structurally undercounted before a full second has elapsed (I4).
        self._history: list[tuple[bool, float | None]] = [
            (True, None) for _ in range(self.window_frames)
        ]

    def update(
        self, frame_index: int, mask: np.ndarray, uncertain: bool = False
    ) -> FlickerScore:
        area_ratio = flagged_area_ratio(mask)
        had_flag = uncertain or area_ratio > 0.0
        self._history.append((had_flag, None if uncertain else area_ratio))
        if len(self._history) > self.window_frames:
            self._history.pop(0)

        flagged_frame_count = sum(1 for flagged, _ in self._history if flagged)
        measured_areas = [area for _, area in self._history if area is not None]
        return FlickerScore(
            frame_index=frame_index,
            flagged_frame_count_last_second=flagged_frame_count,
            flagged_area_ratio=area_ratio,
            max_flagged_area_in_window=max(measured_areas) if measured_areas else 0.0,
            uncertain=len(measured_areas) < len(self._history),
        )
