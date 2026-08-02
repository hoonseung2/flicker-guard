import numpy as np
from detector.scoring import FlickerScore, flagged_area_ratio, WindowedFlashCounter


def test_flagged_area_ratio_half_frame():
    mask = np.zeros((10, 10), dtype=bool)
    mask[:5, :] = True
    assert flagged_area_ratio(mask) == 0.5


def test_flagged_area_ratio_empty_mask():
    mask = np.zeros((10, 10), dtype=bool)
    assert flagged_area_ratio(mask) == 0.0


def test_windowed_counter_counts_flagged_frames_within_one_second_window():
    counter = WindowedFlashCounter(fps=10)  # window = 10 frames
    full_mask = np.ones((2, 2), dtype=bool)
    empty_mask = np.zeros((2, 2), dtype=bool)
    pattern = [full_mask, empty_mask] * 5  # 5 flagged frames across 10 frames
    scores = [counter.update(i, m) for i, m in enumerate(pattern)]
    # 5 flagged FRAMES, which is ~2x the ~2.5 visual flashes they represent.
    assert scores[-1].flagged_frame_count_last_second == 5


def test_windowed_counter_drops_old_frames_outside_window():
    counter = WindowedFlashCounter(fps=4)  # window = 4 frames
    full_mask = np.ones((2, 2), dtype=bool)
    empty_mask = np.zeros((2, 2), dtype=bool)
    for i, m in enumerate([full_mask, empty_mask, full_mask, empty_mask]):
        last = counter.update(i, m)
    assert last.flagged_frame_count_last_second == 2
    # push one more empty frame; window slides, oldest flagged frame drops out
    for i, m in enumerate([empty_mask, empty_mask, empty_mask], start=4):
        last = counter.update(i, m)
    assert last.flagged_frame_count_last_second == 0


def test_flicker_score_carries_windowed_and_instantaneous_area():
    score = FlickerScore(
        frame_index=3,
        flagged_frame_count_last_second=2,
        flagged_area_ratio=0.15,
        max_flagged_area_in_window=0.42,
        uncertain=True,
    )
    assert score.frame_index == 3
    assert score.flagged_frame_count_last_second == 2
    assert score.flagged_area_ratio == 0.15
    assert score.max_flagged_area_in_window == 0.42
    assert score.uncertain is True


def test_windowed_counter_counts_sustained_strobe_correctly():
    counter = WindowedFlashCounter(fps=10)  # window = 10 frames
    full_mask = np.ones((2, 2), dtype=bool)
    scores = [counter.update(i, full_mask) for i in range(15)]  # sustained past the window
    # every frame in the trailing window is flagged -> should report the full window size, not 1
    assert scores[-1].flagged_frame_count_last_second == 10


def test_windowed_counter_remembers_peak_area_across_the_window():
    # C2: the area operand must be windowed, so a duty-cycle strobe's
    # big-area frame keeps counting for the rest of the second.
    counter = WindowedFlashCounter(fps=5)  # window = 5 frames
    big = np.zeros((10, 10), dtype=bool)
    big[:8, :] = True  # 0.8 area
    empty = np.zeros((10, 10), dtype=bool)
    scores = [counter.update(i, m) for i, m in enumerate([big] + [empty] * 4)]
    assert scores[0].flagged_area_ratio == 0.8
    # instantaneous area collapses to 0 immediately...
    assert scores[-1].flagged_area_ratio == 0.0
    # ...but the windowed peak survives until the frame leaves the window
    assert scores[-1].max_flagged_area_in_window == 0.8
    dropped = counter.update(5, empty)
    assert dropped.max_flagged_area_in_window == 0.0


def test_windowed_counter_primes_window_conservatively_before_a_second_elapses():
    # I4: before a full second has elapsed the window is not yet measurable.
    # Those slots must default to flagged, not to safe.
    counter = WindowedFlashCounter(fps=10)
    empty = np.zeros((4, 4), dtype=bool)
    first = counter.update(0, empty)
    # 9 of the window's 10 slots have not elapsed yet: assumed flagged, not safe.
    assert first.flagged_frame_count_last_second == 9
    assert first.uncertain is True
    # once a full second of real frames has been seen, nothing is assumed
    for i in range(1, 10):
        last = counter.update(i, empty)
    assert last.flagged_frame_count_last_second == 0
    assert last.uncertain is False


def test_uncertain_frame_counts_as_flagged_but_contributes_no_measured_area():
    # I4: an unmeasurable frame must push toward risky on the frequency axis
    # without fabricating a full-screen area that would poison the whole window.
    counter = WindowedFlashCounter(fps=3)
    full = np.ones((4, 4), dtype=bool)
    empty = np.zeros((4, 4), dtype=bool)
    score = counter.update(0, full, uncertain=True)
    assert score.flagged_area_ratio == 1.0  # reported as the worst case
    assert score.max_flagged_area_in_window == 0.0  # but not treated as measured
    later = counter.update(1, empty)
    assert later.flagged_frame_count_last_second >= 2  # frame 0 still counted
    assert later.max_flagged_area_in_window == 0.0
