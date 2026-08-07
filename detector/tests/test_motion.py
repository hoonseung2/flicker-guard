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


def test_guard_does_not_fire_on_a_clean_translation():
    # A textured image shifted by a known amount is exactly what phase
    # correlation is for -- the guard must stay out of the way.
    rng = np.random.default_rng(0)
    base = rng.random((128, 128)).astype(np.float32)
    shifted = np.roll(base, shift=5, axis=1)

    dx, dy = estimate_global_shift(base, shifted)
    assert abs(dx - 5.0) < 1.0
    assert abs(dy) < 1.0


def test_guard_rejects_a_shift_between_unrelated_noise_frames():
    # Independent noise has no correspondence, so any shift phase
    # correlation reports is fabricated. Returning zero means "no
    # compensation", which is what the pipeline did before the motion stage
    # existed and never compares unrelated pixels.
    rng = np.random.default_rng(1)
    a = rng.random((128, 128)).astype(np.float32)
    b = rng.random((128, 128)).astype(np.float32)

    assert estimate_global_shift(a, b) == (0.0, 0.0)


def test_guard_rejects_a_shift_when_there_is_almost_no_structure():
    # A nearly flat frame gives phase correlation nothing to lock onto, so
    # whatever shift it reports comes from the noise floor. This is the same
    # failure class found on real footage -- clip 57523 scores 0.0098 on
    # every one of its 299 frame pairs, despite the mildest flicker in the
    # set and no scene cuts.
    rng = np.random.default_rng(3)
    flat_a = np.full((128, 128), 0.5, dtype=np.float32) + rng.random((128, 128)).astype(np.float32) * 1e-4
    flat_b = np.full((128, 128), 0.5, dtype=np.float32) + rng.random((128, 128)).astype(np.float32) * 1e-4

    assert estimate_global_shift(flat_a, flat_b) == (0.0, 0.0)


def test_min_response_zero_restores_the_unguarded_behaviour():
    # Callers that want the raw estimate (e.g. a diagnostic that plots the
    # distribution) must be able to opt out without reimplementing it.
    rng = np.random.default_rng(2)
    a = rng.random((128, 128)).astype(np.float32)
    b = rng.random((128, 128)).astype(np.float32)

    assert estimate_global_shift(a, b, min_response=0.0) != (0.0, 0.0)


def test_min_response_zero_passes_through_a_negative_response(monkeypatch):
    # None of the fixture seeds above produce a negative phaseCorrelate
    # response (seeds 1, 2, 3 measure 0.0319, 0.0059, 0.0016 -- all
    # positive), yet real footage does: the measurement that motivated this
    # guard recorded -0.000. `response < min_response` with min_response=0.0
    # would still fire on a negative response and silently filter it, which
    # would censor the distribution a later diagnostic (Task 5) needs to see
    # in full. Monkeypatch cv2.phaseCorrelate to force a negative response,
    # since no seed hunt can guarantee one.
    import detector.motion as motion

    def fake_phase_correlate(prev, curr):
        return (7.0, -3.0), -0.5

    monkeypatch.setattr(motion.cv2, "phaseCorrelate", fake_phase_correlate)

    prev = np.zeros((128, 128), dtype=np.float32)
    curr = np.zeros((128, 128), dtype=np.float32)

    assert estimate_global_shift(prev, curr, min_response=0.0) == (7.0, -3.0)
