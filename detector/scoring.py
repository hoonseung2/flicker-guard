"""Aggregates per-pixel transition masks into a windowed flash-frequency and
area score. Flash frequency is approximated as the count of frames with a
qualifying transition within a trailing 1-second window — an MVP simplification
that treats every transition-bearing frame as one flash event, which is
conservative (may overcount slow single flashes as multiple events) but never
fails to flag a sustained strobe, unlike edge-counting."""
from dataclasses import dataclass

import numpy as np


@dataclass
class FlickerScore:
    frame_index: int
    flash_count_last_second: int
    flagged_area_ratio: float


def flagged_area_ratio(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask)) / mask.size


class WindowedFlashCounter:
    def __init__(self, fps: float):
        self.fps = fps
        self.window_frames = max(1, round(fps))
        self._had_any_flag_history: list[bool] = []

    def update(self, frame_index: int, mask: np.ndarray) -> FlickerScore:
        area_ratio = flagged_area_ratio(mask)
        had_flag = area_ratio > 0.0
        self._had_any_flag_history.append(had_flag)
        if len(self._had_any_flag_history) > self.window_frames:
            self._had_any_flag_history.pop(0)
        flash_count = sum(self._had_any_flag_history)
        return FlickerScore(
            frame_index=frame_index,
            flash_count_last_second=flash_count,
            flagged_area_ratio=area_ratio,
        )
