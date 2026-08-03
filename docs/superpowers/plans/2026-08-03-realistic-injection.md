# Realistic Flicker Injection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a linear-light-space multiplicative-gain flicker injection alongside DatasetSynth's existing flat-color-overwrite injection, so synthesized training data preserves the underlying scene's texture instead of erasing it, and wire it into the batch CLI as a selectable mode.

**Architecture:** Three additive changes to three existing, already-shipped files — no existing function's behavior changes. `training/injection.py` gains two new pixel-space functions (`inject_general_flash_realistic`, `inject_red_flash_realistic`) built on two new private sRGB↔linear array helpers. `training/synth.py` gains `synthesize_sample_realistic`, which reuses `sample_synthesis_params` for window placement and the existing Detector-validation retry loop, swapping only which injection function runs. `training/cli.py` gains an `--injection-mode {flat,realistic}` flag so the batch tool can actually produce either kind of dataset.

**Tech Stack:** Python 3.10+, numpy. No new dependencies.

## Global Constraints

- Every addition in this plan is purely additive: `inject_general_flash`, `inject_red_flash`, `sample_synthesis_params`, `synthesize_sample`, and `run_batch`'s existing default behavior (`injection_mode="flat"`) must not change in any way — all existing tests in `training/tests/` must keep passing unmodified.
- Gain modulation happens in **linear light space** (sRGB decoded, gain multiplied, re-encoded), not directly on sRGB values — this is the physically accurate model for a light source's intensity changing, per the design spec.
- No offset/floor added for near-black pixels that don't cross the threshold under any gain — this is accepted as physically realistic, not a bug to work around.
- Gain values are **fixed constants**, not computed per-profile or content-adaptive — validated empirically to work broadly (see spec §3), reusing the existing retry-and-skip infrastructure for any content that still fails.
- No new dependencies beyond numpy.
- Tests use only small in-memory synthetic frames — no real DAVIS data required to run `pytest`. Tests must use content with genuine spatial variation (not solid single-value frames) at least once per new injection function, to prove texture is actually preserved.

---

## File Structure

```
flicker-guard/
├── training/
│   ├── injection.py    # Task 1 (modified — additive only)
│   ├── synth.py         # Task 2 (modified — additive only)
│   ├── cli.py            # Task 3 (modified — additive only)
│   └── tests/
│       ├── test_injection.py   # Task 1 (add tests)
│       ├── test_synth.py        # Task 2 (add tests)
│       └── test_cli.py           # Task 3 (add tests)
```

---

## Task 1: Realistic pixel-space injection functions

**Files:**
- Modify: `flicker-guard/training/injection.py`
- Test: `flicker-guard/training/tests/test_injection.py` (add tests, keep the 2 existing ones unmodified)

**Interfaces:**
- Consumes: `training.params.InjectionWindow` (existing, unchanged).
- Produces: `inject_general_flash_realistic(frames: list[np.ndarray], window: InjectionWindow, gain_dark: float = 0.3, gain_bright: float = 3.0) -> list[np.ndarray]`, `inject_red_flash_realistic(frames: list[np.ndarray], window: InjectionWindow, red_gains: tuple[float, float, float] = (20.0, 0.02, 0.02), baseline_gains: tuple[float, float, float] = (0.3, 1.0, 1.0)) -> list[np.ndarray]`. Both return a **new** list; input `frames` and its arrays are never mutated (same contract as the existing `inject_general_flash`/`inject_red_flash`).

- [ ] **Step 1: Write the failing test**

Add these to `flicker-guard/training/tests/test_injection.py`, alongside the existing 2 tests and `_solid_frames` fixture (do not remove or modify anything already there):

```python
# add to flicker-guard/training/tests/test_injection.py
from training.injection import (
    inject_general_flash,
    inject_general_flash_realistic,
    inject_red_flash,
    inject_red_flash_realistic,
)


def test_inject_general_flash_realistic_leaves_frames_outside_window_untouched():
    frames = _solid_frames(10, 20, 20, 0.5)
    window = InjectionWindow(start_frame=3, end_frame=6, mask_top=0, mask_left=0, mask_height=20, mask_width=20, period_frames=1, ramp_frames=1)
    out = inject_general_flash_realistic(frames, window)
    assert np.array_equal(out[0], frames[0])
    assert np.array_equal(out[9], frames[9])


def test_inject_general_flash_realistic_leaves_pixels_outside_mask_untouched():
    frames = _solid_frames(6, 20, 20, 0.5)
    window = InjectionWindow(start_frame=1, end_frame=4, mask_top=5, mask_left=5, mask_height=5, mask_width=5, period_frames=1, ramp_frames=1)
    out = inject_general_flash_realistic(frames, window)
    assert np.array_equal(out[2][0:5, :, :], frames[2][0:5, :, :])
    assert frames[2][0, 0, 0] == 0.5  # input untouched


def test_inject_general_flash_realistic_preserves_texture():
    # Unlike inject_general_flash (flat overwrite), the two halves must stay
    # visibly different from each other after injection -- a flat overwrite
    # would collapse both to the same value.
    frame = np.zeros((10, 10, 3), dtype=np.float32)
    frame[0:5, :, :] = 0.2
    frame[5:10, :, :] = 0.6
    frames = [frame.copy() for _ in range(6)]
    window = InjectionWindow(start_frame=1, end_frame=4, mask_top=0, mask_left=0, mask_height=10, mask_width=10, period_frames=1, ramp_frames=1)
    out = inject_general_flash_realistic(frames, window, gain_dark=0.3, gain_bright=3.0)
    assert not np.allclose(out[2][0:5], out[2][5:10])


def test_inject_general_flash_realistic_transitions_are_flagged_by_detector():
    from detector.flash import transition_mask
    from detector.luminance import relative_luminance

    rng = np.random.default_rng(0)
    frame = rng.uniform(0.1, 0.9, size=(20, 20, 3)).astype(np.float32)
    frames = [frame.copy() for _ in range(8)]
    window = InjectionWindow(start_frame=1, end_frame=6, mask_top=0, mask_left=0, mask_height=20, mask_width=20, period_frames=1, ramp_frames=1)
    out = inject_general_flash_realistic(frames, window, gain_dark=0.3, gain_bright=3.0)
    lum2 = relative_luminance(out[2])
    lum3 = relative_luminance(out[3])
    mask = transition_mask(lum2, lum3, dark_threshold=0.80, delta_threshold=0.10)
    assert mask.mean() > 0.9  # measured 0.9875 on this exact fixture


def test_inject_red_flash_realistic_transitions_are_flagged_by_detector():
    from detector.flash import red_flash_mask

    rng = np.random.default_rng(0)
    frame = rng.uniform(0.1, 0.9, size=(10, 10, 3)).astype(np.float32)
    frames = [frame.copy() for _ in range(6)]
    window = InjectionWindow(start_frame=1, end_frame=4, mask_top=0, mask_left=0, mask_height=10, mask_width=10, period_frames=1, ramp_frames=1)
    out = inject_red_flash_realistic(frames, window)
    mask = red_flash_mask(out[2], out[3], saturation_ratio_threshold=0.80)
    assert mask.mean() > 0.9  # measured 0.95 on this exact fixture
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest training/tests/test_injection.py -v`
Expected: the 5 new tests FAIL with `ImportError: cannot import name 'inject_general_flash_realistic'`; the 2 pre-existing tests still PASS.

- [ ] **Step 3: Add the realistic injection functions**

Append to `flicker-guard/training/injection.py` (do not modify anything above this point in the file):

```python
def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Array version of the sRGB decode -- the same formula
    `_gray_channel_for_luminance` inverts for the scalar/gray case, kept as a
    separate function (not shared with it) so neither has to change to
    support the other."""
    low = c <= 0.04045
    return np.where(low, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    low = c <= _LINEAR_CUTOFF
    return np.where(low, c * 12.92, (1 + _SRGB_A) * (c ** (1 / 2.4)) - _SRGB_A)


def inject_general_flash_realistic(
    frames: list[np.ndarray],
    window: InjectionWindow,
    gain_dark: float = 0.3,
    gain_bright: float = 3.0,
) -> list[np.ndarray]:
    """Alternates the masked region between two *multiplicative* luminance
    gains in linear light space, instead of inject_general_flash's flat
    color overwrite -- this preserves the underlying scene's texture, matching
    how a real light source's intensity change multiplies reflected light
    physically. Gain defaults validated empirically (see plan/spec) to
    reliably cross Detector's default thresholds on real footage."""
    rows, cols = _mask_slice(window)
    out = [frame.copy() for frame in frames]
    for i in range(window.start_frame, window.end_frame + 1):
        offset = i - window.start_frame
        gain = gain_bright if _is_pulse_frame(offset, window.period_frames) else gain_dark
        region = out[i][rows, cols, :]
        linear = _srgb_to_linear(region)
        out[i][rows, cols, :] = _linear_to_srgb(linear * gain)
    return out


def inject_red_flash_realistic(
    frames: list[np.ndarray],
    window: InjectionWindow,
    red_gains: tuple[float, float, float] = (20.0, 0.02, 0.02),
    baseline_gains: tuple[float, float, float] = (0.3, 1.0, 1.0),
) -> list[np.ndarray]:
    """Alternates the masked region between two per-channel gain triples in
    linear light space -- red_gains boosts R and suppresses G/B, baseline_gains
    suppresses R -- instead of inject_red_flash's flat saturated-color
    overwrite. Both states are deliberately engineered (neither is a pass-
    through of the original), because a pass-through baseline can itself
    exceed the saturation-ratio threshold on already-reddish content,
    breaking the XOR-based transition Detector relies on. gain magnitudes are
    large because the WCAG saturation-ratio test runs in gamma-compressed
    sRGB space, which compresses a 10x linear channel-ratio difference down
    to roughly 2.6x -- verified empirically (see plan/spec)."""
    rows, cols = _mask_slice(window)
    out = [frame.copy() for frame in frames]
    for i in range(window.start_frame, window.end_frame + 1):
        offset = i - window.start_frame
        gains = red_gains if _is_pulse_frame(offset, window.period_frames) else baseline_gains
        region = out[i][rows, cols, :]
        linear = _srgb_to_linear(region)
        modulated = linear * np.array(gains, dtype=np.float32)
        out[i][rows, cols, :] = _linear_to_srgb(modulated)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest training/tests/test_injection.py -v`
Expected: PASS (7 passed — 2 pre-existing + 5 new)

- [ ] **Step 5: Commit**

```bash
git add flicker-guard/training/injection.py flicker-guard/training/tests/test_injection.py
git commit -m "feat(dataset-synth): add linear-space realistic flash injection"
```

---

## Task 2: `synthesize_sample_realistic`

**Files:**
- Modify: `flicker-guard/training/synth.py`
- Test: `flicker-guard/training/tests/test_synth.py` (add tests, keep existing ones unmodified)

**Interfaces:**
- Consumes: `inject_general_flash_realistic`, `inject_red_flash_realistic` (Task 1); `sample_synthesis_params`, `SynthParams`, `InjectionWindow` (existing, unchanged); `run_detection` (existing, unchanged); `SynthesizedSample`, `_window_is_covered` (existing, unchanged, reused directly).
- Produces: `synthesize_sample_realistic(clean_frames: list[np.ndarray], fps: float, profile: ThresholdProfile, pattern: str, rng: np.random.Generator, max_retries: int = 5) -> SynthesizedSample | None` — same contract as the existing `synthesize_sample`.

- [ ] **Step 1: Write the failing test**

Add these to `flicker-guard/training/tests/test_synth.py`, alongside the existing tests, `PROFILE`, and `_clean_clip` (do not remove or modify anything already there):

```python
# add to flicker-guard/training/tests/test_synth.py
from training.synth import synthesize_sample_realistic


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
    sample = synthesize_sample_realistic(_textured_clip(seed=1), fps=30.0, profile=PROFILE, pattern="general", rng=rng)
    assert sample is not None
    mid_frame_idx = (sample.window.start_frame + sample.window.end_frame) // 2
    degraded = sample.degraded_frames[mid_frame_idx]
    top, left = sample.window.mask_top, sample.window.mask_left
    h, w = sample.window.mask_height, sample.window.mask_width
    region = degraded[top:top + h, left:left + w]
    # the original base frame has per-pixel spatial variation; a flat-color
    # injection (the old inject_general_flash) would collapse this region to
    # one constant value with std == 0
    assert region.std() > 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest training/tests/test_synth.py -v`
Expected: the 3 new tests FAIL with `ImportError: cannot import name 'synthesize_sample_realistic'`; existing tests still PASS.

- [ ] **Step 3: Add `synthesize_sample_realistic`**

Two changes to `flicker-guard/training/synth.py`. First, change the existing `training.injection` import line from:

```python
from training.injection import inject_general_flash, inject_red_flash
```

to:

```python
from training.injection import (
    inject_general_flash,
    inject_general_flash_realistic,
    inject_red_flash,
    inject_red_flash_realistic,
)
```

That is the only change to any existing line in the file — do not touch `synthesize_sample`, `_apply_pattern`, `_window_is_covered`, or `SynthesizedSample`. Second, append at the end of the file:

```python
def _apply_pattern_realistic(clean_frames: list[np.ndarray], params: SynthParams) -> list[np.ndarray]:
    if params.pattern == "general":
        return inject_general_flash_realistic(clean_frames, params.window)
    return inject_red_flash_realistic(clean_frames, params.window)


def synthesize_sample_realistic(
    clean_frames: list[np.ndarray],
    fps: float,
    profile: ThresholdProfile,
    pattern: str,
    rng: np.random.Generator,
    max_retries: int = 5,
) -> SynthesizedSample | None:
    """Same retry-and-validate structure as synthesize_sample, but injects
    with the linear-space gain functions instead of the flat-color-overwrite
    ones. sample_synthesis_params is reused only for its InjectionWindow
    (position/size/timing) -- its dark_target/bright_target/red_rgb/
    baseline_rgb fields are computed but unused here, since the realistic
    path uses fixed gain constants (see training.injection) rather than
    per-profile absolute targets."""
    frame_height, frame_width = clean_frames[0].shape[:2]

    for _ in range(max_retries):
        params = sample_synthesis_params(
            profile, pattern, len(clean_frames), frame_height, frame_width, fps, rng
        )
        degraded_frames = _apply_pattern_realistic(clean_frames, params)
        _, segments = run_detection(degraded_frames, fps=fps, profile=profile, margin_seconds=0.0)
        if _window_is_covered(params.window, segments):
            return SynthesizedSample(
                pattern=pattern,
                profile_name=profile.name,
                window=params.window,
                clean_frames=clean_frames,
                degraded_frames=degraded_frames,
                segments=segments,
            )
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest training/tests/test_synth.py -v`
Expected: PASS (7 passed — 4 pre-existing + 3 new)

- [ ] **Step 5: Commit**

```bash
git add flicker-guard/training/synth.py flicker-guard/training/tests/test_synth.py
git commit -m "feat(dataset-synth): add synthesize_sample_realistic"
```

---

## Task 3: `--injection-mode` CLI flag

**Files:**
- Modify: `flicker-guard/training/cli.py`
- Test: `flicker-guard/training/tests/test_cli.py` (add tests, keep existing ones unmodified)

**Interfaces:**
- Consumes: `synthesize_sample`, `synthesize_sample_realistic` (existing + Task 2, both imported by name so both are independently monkeypatchable in tests).
- Produces: `run_batch(clips_dir, profiles_dir, out_root, samples_per_combo, seed=0, overwrite=False, injection_mode="flat") -> dict` (one new keyword parameter, default preserves existing behavior exactly), `main(argv=None) -> int` (gains `--injection-mode {flat,realistic}`, default `flat`).

- [ ] **Step 1: Write the failing test**

Add these to `flicker-guard/training/tests/test_cli.py`, alongside the existing tests, `PROFILE_JSON`, `_write_clip`, `_write_profiles_dir` (do not remove or modify anything already there):

```python
# add to flicker-guard/training/tests/test_cli.py
def test_run_batch_realistic_mode_calls_synthesize_sample_realistic(tmp_path, monkeypatch):
    import training.cli as cli_module

    calls = []

    def _fake_realistic(*args, **kwargs):
        calls.append(1)
        return None

    monkeypatch.setattr(cli_module, "synthesize_sample_realistic", _fake_realistic)

    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _write_clip(clips_dir / "bear.mp4")
    profiles_dir = tmp_path / "profiles"
    _write_profiles_dir(profiles_dir)
    out_dir = tmp_path / "out"

    summary = cli_module.run_batch(
        clips_dir, profiles_dir, out_dir, samples_per_combo=1, injection_mode="realistic"
    )

    assert len(calls) == 2  # one call per pattern (general, red)
    assert summary["failed"] == 2  # the fake always returns None -> validation_exhausted


def test_run_batch_defaults_to_flat_injection_mode(tmp_path, monkeypatch):
    import training.cli as cli_module

    calls = []

    def _fake_flat(*args, **kwargs):
        calls.append(1)
        return None

    monkeypatch.setattr(cli_module, "synthesize_sample", _fake_flat)

    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _write_clip(clips_dir / "bear.mp4")
    profiles_dir = tmp_path / "profiles"
    _write_profiles_dir(profiles_dir)
    out_dir = tmp_path / "out"

    cli_module.run_batch(clips_dir, profiles_dir, out_dir, samples_per_combo=1)  # no injection_mode passed

    assert len(calls) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest training/tests/test_cli.py -v`
Expected: the 2 new tests FAIL (`test_run_batch_realistic_mode_calls_synthesize_sample_realistic` fails because `run_batch` doesn't accept `injection_mode` yet — `TypeError: run_batch() got an unexpected keyword argument 'injection_mode'`); existing tests still PASS.

- [ ] **Step 3: Add the `--injection-mode` flag**

In `flicker-guard/training/cli.py`, change the existing import line from:

```python
from training.synth import synthesize_sample
```

to:

```python
from training.synth import synthesize_sample, synthesize_sample_realistic
```

Change the `run_batch` signature and the one line inside it that calls `synthesize_sample`. The signature becomes:

```python
def run_batch(
    clips_dir: Path,
    profiles_dir: Path,
    out_root: Path,
    samples_per_combo: int,
    seed: int = 0,
    overwrite: bool = False,
    injection_mode: str = "flat",
) -> dict:
```

Immediately inside the function body, before the existing `profiles = _load_profiles(profiles_dir)` line, add:

```python
    if injection_mode == "flat":
        synthesize = synthesize_sample
    elif injection_mode == "realistic":
        synthesize = synthesize_sample_realistic
    else:
        raise ValueError(f"unknown injection_mode: {injection_mode!r}")

```

Then change the existing call site inside the retry loop from:

```python
                    try:
                        sample = synthesize_sample(
                            clean_frames, fps, profile, pattern, _sample_rng(seed, sid)
                        )
```

to:

```python
                    try:
                        sample = synthesize(
                            clean_frames, fps, profile, pattern, _sample_rng(seed, sid)
                        )
```

In `main()`, add the new argument (anywhere among the other `parser.add_argument` calls) and pass it through:

```python
    parser.add_argument(
        "--injection-mode",
        choices=["flat", "realistic"],
        default="flat",
        help=(
            "flat: overwrite pixels with a fixed color (original DatasetSynth "
            "behavior). realistic: multiply original luminance/color by a fixed "
            "gain in linear light space, preserving scene texture."
        ),
    )
```

and add `injection_mode=args.injection_mode,` to the existing `run_batch(...)` call inside `main()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest training/tests/test_cli.py -v`
Expected: PASS (6 passed — 4 pre-existing + 2 new)

Then run the full repo suite to confirm nothing else broke:
Run: `pytest -v` (from the flicker-guard repo root)
Expected: PASS (135 passed — 125 pre-existing + 5 (Task 1) + 3 (Task 2) + 2 (Task 3) = 135)

- [ ] **Step 5: Commit**

```bash
git add flicker-guard/training/cli.py flicker-guard/training/tests/test_cli.py
git commit -m "feat(dataset-synth): add --injection-mode flag to the batch CLI"
```
