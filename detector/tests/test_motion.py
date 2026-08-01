import numpy as np
from detector.motion import estimate_global_shift, compensate_shift


def _random_texture(h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((h, w)).astype(np.float32)


def test_estimate_global_shift_recovers_known_pan():
    prev = _random_texture()
    curr = np.roll(prev, shift=(3, 5), axis=(0, 1))  # shift 5 px x, 3 px y
    dx, dy = estimate_global_shift(prev, curr)
    assert abs(dx - 5) < 1.0
    assert abs(dy - 3) < 1.0


def test_compensate_shift_removes_pan_within_tolerance():
    prev = _random_texture()
    curr = np.roll(prev, shift=(3, 5), axis=(0, 1))
    dx, dy = estimate_global_shift(prev, curr)
    aligned = compensate_shift(curr, -dx, -dy)
    interior = slice(16, 48)
    diff = np.abs(aligned[interior, interior] - prev[interior, interior])
    assert diff.mean() < 0.05
