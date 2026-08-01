import numpy as np
from detector.flash import transition_mask, red_flash_mask

def test_transition_mask_flags_large_dark_to_bright_change():
    prev = np.full((2, 2), 0.10, dtype=np.float32)
    curr = np.full((2, 2), 0.90, dtype=np.float32)
    mask = transition_mask(prev, curr)
    assert mask.all()

def test_transition_mask_ignores_small_change():
    prev = np.full((2, 2), 0.50, dtype=np.float32)
    curr = np.full((2, 2), 0.55, dtype=np.float32)  # delta 0.05 < 0.10 threshold
    mask = transition_mask(prev, curr)
    assert not mask.any()

def test_transition_mask_ignores_bright_to_brighter_change():
    prev = np.full((2, 2), 0.85, dtype=np.float32)
    curr = np.full((2, 2), 0.99, dtype=np.float32)  # both above 0.80 dark threshold
    mask = transition_mask(prev, curr)
    assert not mask.any()

def test_transition_mask_boundary_exactly_at_threshold_is_not_flagged():
    prev = np.full((2, 2), 0.40, dtype=np.float32)
    curr = np.full((2, 2), 0.50, dtype=np.float32)  # delta exactly 0.10, needs to exceed
    mask = transition_mask(prev, curr)
    assert not mask.any()

def test_red_flash_mask_flags_saturated_red_onset():
    prev = np.zeros((1, 1, 3), dtype=np.float32)
    curr = np.zeros((1, 1, 3), dtype=np.float32)
    curr[0, 0] = [0.9, 0.05, 0.05]  # saturated red
    mask = red_flash_mask(prev, curr)
    assert mask[0, 0]

def test_red_flash_mask_ignores_non_saturated_colors():
    prev = np.zeros((1, 1, 3), dtype=np.float32)
    curr = np.zeros((1, 1, 3), dtype=np.float32)
    curr[0, 0] = [0.5, 0.5, 0.5]  # gray, not red
    mask = red_flash_mask(prev, curr)
    assert not mask[0, 0]
