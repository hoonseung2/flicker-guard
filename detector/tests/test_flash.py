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

def test_transition_mask_honours_caller_supplied_thresholds():
    # I3: thresholds are parameters, not module constants. With a stricter
    # delta threshold the same transition must stop being flagged.
    prev = np.full((2, 2), 0.10, dtype=np.float32)
    curr = np.full((2, 2), 0.30, dtype=np.float32)  # delta 0.20
    assert transition_mask(prev, curr).all()
    assert not transition_mask(prev, curr, delta_threshold=0.50).any()
    # ...and a lower dark threshold likewise excludes it.
    assert not transition_mask(prev, curr, dark_threshold=0.05).any()

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

def test_red_flash_mask_flags_mid_dark_saturated_red():
    # I1: WCAG defines saturated red as R/(R+G+B) >= 0.8. (0.60, 0.05, 0.05)
    # has ratio 0.857 -> saturated, even though R is below the old 0.8 box cut.
    prev = np.zeros((1, 1, 3), dtype=np.float32)
    curr = np.zeros((1, 1, 3), dtype=np.float32)
    curr[0, 0] = [0.60, 0.05, 0.05]
    mask = red_flash_mask(prev, curr)
    assert mask[0, 0]

def test_red_flash_mask_treats_pure_black_as_not_saturated_red():
    # R+G+B == 0 must not divide by zero and must not count as red.
    prev = np.zeros((2, 2, 3), dtype=np.float32)
    curr = np.zeros((2, 2, 3), dtype=np.float32)
    with np.errstate(invalid="raise", divide="raise"):
        mask = red_flash_mask(prev, curr)
    assert not mask.any()

def test_red_flash_mask_ignores_bright_red_below_ratio_threshold():
    # R is high but so are G/B -> ratio 0.45, not a WCAG saturated red.
    prev = np.zeros((1, 1, 3), dtype=np.float32)
    curr = np.zeros((1, 1, 3), dtype=np.float32)
    curr[0, 0] = [0.9, 0.55, 0.55]
    assert not red_flash_mask(prev, curr)[0, 0]

def test_red_flash_mask_honours_caller_supplied_ratio_threshold():
    prev = np.zeros((1, 1, 3), dtype=np.float32)
    curr = np.zeros((1, 1, 3), dtype=np.float32)
    curr[0, 0] = [0.60, 0.05, 0.05]  # ratio 0.857
    assert red_flash_mask(prev, curr, saturation_ratio_threshold=0.80)[0, 0]
    assert not red_flash_mask(prev, curr, saturation_ratio_threshold=0.95)[0, 0]
