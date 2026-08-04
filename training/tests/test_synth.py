import numpy as np

from detector.profiles import ThresholdProfile
from training.synth import synthesize_sample, synthesize_sample_realistic

PROFILE = ThresholdProfile(
    name="test", max_flashes_per_second=3, max_area_ratio=0.10,
    general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
    red_saturation_ratio_threshold=0.80,
)


def _clean_clip(n=90, h=40, w=40, value=0.5):
    return [np.full((h, w, 3), value, dtype=np.float32) for _ in range(n)]


def test_synthesize_sample_accepts_general_flash_and_covers_steady_state_tail():
    rng = np.random.default_rng(0)
    sample = synthesize_sample(_clean_clip(), fps=30.0, profile=PROFILE, pattern="general", rng=rng)
    assert sample is not None
    assert sample.pattern == "general"
    assert len(sample.degraded_frames) == 90
    core_start = sample.window.start_frame + sample.window.ramp_frames
    assert any(
        seg.start_frame <= core_start and seg.end_frame >= sample.window.end_frame
        for seg in sample.segments
    )


def test_synthesize_sample_accepts_red_flash():
    rng = np.random.default_rng(1)
    sample = synthesize_sample(_clean_clip(), fps=30.0, profile=PROFILE, pattern="red", rng=rng)
    assert sample is not None
    assert sample.pattern == "red"


def test_synthesize_sample_leaves_clean_frames_unmodified_and_degraded_differs():
    clean = _clean_clip()
    rng = np.random.default_rng(0)
    sample = synthesize_sample(clean, fps=30.0, profile=PROFILE, pattern="general", rng=rng)
    for original, clean_out in zip(clean, sample.clean_frames):
        assert np.array_equal(original, clean_out)
    probe = sample.window.start_frame + 1
    assert not np.array_equal(sample.degraded_frames[probe], clean[probe])


def test_synthesize_sample_returns_none_after_exhausting_retries(monkeypatch):
    import training.synth as synth_module

    def _always_empty(*args, **kwargs):
        return [], []

    calls = []
    original_sample_params = synth_module.sample_synthesis_params

    def _counting_sample_params(*args, **kwargs):
        calls.append(1)
        return original_sample_params(*args, **kwargs)

    monkeypatch.setattr(synth_module, "run_detection", _always_empty)
    monkeypatch.setattr(synth_module, "sample_synthesis_params", _counting_sample_params)

    rng = np.random.default_rng(0)
    sample = synthesize_sample(_clean_clip(), fps=30.0, profile=PROFILE, pattern="general", rng=rng, max_retries=3)

    assert sample is None
    assert len(calls) == 3


def _textured_clip(n=90, h=40, w=40, seed=0):
    # A single frame with genuine spatial variation, replicated across all
    # timesteps -- unlike _clean_clip's single flat value, this lets tests
    # confirm the realistic injection preserves per-pixel differences, while
    # still being temporally coherent (same content every frame) like real
    # clean footage, not independent noise per frame.
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.1, 0.9, size=(h, w, 3)).astype(np.float32)
    return [base.copy() for _ in range(n)]


def test_synthesize_sample_realistic_accepts_general_flash():
    rng = np.random.default_rng(0)
    sample = synthesize_sample_realistic(_textured_clip(), fps=30.0, profile=PROFILE, pattern="general", rng=rng)
    assert sample is not None
    assert sample.pattern == "general"
    assert len(sample.degraded_frames) == 90


def test_synthesize_sample_realistic_accepts_red_flash():
    rng = np.random.default_rng(1)
    sample = synthesize_sample_realistic(_textured_clip(), fps=30.0, profile=PROFILE, pattern="red", rng=rng)
    assert sample is not None
    assert sample.pattern == "red"


def test_synthesize_sample_realistic_preserves_spatial_texture_in_degraded_frames():
    rng = np.random.default_rng(0)
    clean = _textured_clip(seed=1)
    sample = synthesize_sample_realistic(clean, fps=30.0, profile=PROFILE, pattern="general", rng=rng)
    assert sample is not None
    mid_frame_idx = (sample.window.start_frame + sample.window.end_frame) // 2
    degraded = sample.degraded_frames[mid_frame_idx]
    original = clean[mid_frame_idx]
    changed = ~np.isclose(degraded, original).all(axis=-1)
    assert changed.any(), "expected some pixels to change in the injected frame"
    # the original base frame has per-pixel spatial variation; a flat-color
    # injection (the old inject_general_flash) would collapse the changed
    # region to one constant value with std == 0
    assert degraded[changed].std() > 0.01


def test_synthesize_sample_realistic_accepts_general_flash_with_lights():
    rng = np.random.default_rng(0)
    sample = synthesize_sample_realistic(_textured_clip(), fps=30.0, profile=PROFILE, pattern="general", rng=rng)
    assert sample is not None
    assert 1 <= len(sample.window.lights) <= 3


def test_synthesize_sample_realistic_exercises_full_runway_branch_with_lights():
    # Every other test in this file uses n=90, fps=30.0: duration_frames=42
    # (window_frames=30, period_frames=3, 30+4*3=42) -> latest_start=90-42=48,
    # which is below desired_runway=2*window_frames=60, so
    # sample_synthesis_params always squeezes start_frame to the single
    # degenerate value 48. That means Task 1's "clip is long enough, pick
    # anywhere in the runway range" placement branch has never actually run
    # together with real light sampling (Task 3) and real Detector
    # re-validation. A 150-frame clip gives latest_start=150-42=108, which is
    # >= desired_runway=60, so start_frame is drawn from the genuine range
    # [60, 108] instead of being squeezed -- every draw must land > 48.
    rng = np.random.default_rng(0)
    sample = synthesize_sample_realistic(
        _textured_clip(n=150), fps=30.0, profile=PROFILE, pattern="general", rng=rng
    )
    assert sample is not None
    assert sample.window.start_frame > 48
