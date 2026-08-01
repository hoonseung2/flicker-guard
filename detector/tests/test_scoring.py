import numpy as np
from detector.scoring import FlickerScore, flagged_area_ratio, WindowedFlashCounter


def test_flagged_area_ratio_half_frame():
    mask = np.zeros((10, 10), dtype=bool)
    mask[:5, :] = True
    assert flagged_area_ratio(mask) == 0.5


def test_flagged_area_ratio_empty_mask():
    mask = np.zeros((10, 10), dtype=bool)
    assert flagged_area_ratio(mask) == 0.0


def test_windowed_counter_counts_onsets_within_one_second_window():
    counter = WindowedFlashCounter(fps=10)  # window = 10 frames
    full_mask = np.ones((2, 2), dtype=bool)
    empty_mask = np.zeros((2, 2), dtype=bool)
    pattern = [full_mask, empty_mask] * 5  # 5 onsets across 10 frames
    scores = [counter.update(i, m) for i, m in enumerate(pattern)]
    assert scores[-1].flash_count_last_second == 5


def test_windowed_counter_drops_old_frames_outside_window():
    counter = WindowedFlashCounter(fps=4)  # window = 4 frames
    full_mask = np.ones((2, 2), dtype=bool)
    empty_mask = np.zeros((2, 2), dtype=bool)
    for i, m in enumerate([full_mask, empty_mask, full_mask, empty_mask]):
        last = counter.update(i, m)
    assert last.flash_count_last_second == 2
    # push one more empty frame; window slides, oldest 'full' onset should drop out
    for i, m in enumerate([empty_mask, empty_mask, empty_mask], start=4):
        last = counter.update(i, m)
    assert last.flash_count_last_second == 0


def test_flicker_score_fields():
    score = FlickerScore(frame_index=3, flash_count_last_second=2, flagged_area_ratio=0.15)
    assert score.frame_index == 3
    assert score.flash_count_last_second == 2
    assert score.flagged_area_ratio == 0.15
