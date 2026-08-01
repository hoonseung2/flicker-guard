"""Aggregates per-pixel transition masks into a windowed flash-frequency and
area score. Flash frequency is approximated as the count of onset edges
(mask goes from empty to non-empty) inside a trailing 1-second window — an
MVP simplification of full opposing-transition-pair counting."""
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
        flash_count = self._count_onsets(self._had_any_flag_history)
        return FlickerScore(
            frame_index=frame_index,
            flash_count_last_second=flash_count,
            flagged_area_ratio=area_ratio,
        )

    @staticmethod
    def _count_onsets(history: list[bool]) -> int:
        count = 0
        prev = False
        for cur in history:
            if cur and not prev:
                count += 1
            prev = cur
        return count
