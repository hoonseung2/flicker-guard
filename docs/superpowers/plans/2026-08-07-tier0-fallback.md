# Tier 0 Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a classical, always-passing PSE mitigation path — compress each frame's contrast toward a trailing reference, at the minimum strength that clears the Detector's area criterion.

**Architecture:** Four small modules under a new `fallback/` package. `transfer.py` holds the pure per-frame affine compression in linear light; `strength.py` computes the required strength analytically from the motion-compensated luminance-delta distribution; `apply.py` applies it per risk segment with a raised-cosine fade and runs the verify/escalate loop; `cli.py` wraps it as video-in/video-out. The whole path is deterministic classical signal processing — no model, no GPU.

**Tech Stack:** Python 3.12, NumPy, OpenCV (`cv2`), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-07-tier0-fallback-design.md`

## Global Constraints

- Frames are RGB `float32` in `[0, 1]`, shape `(H, W, 3)` — the convention used by `detector.cli.read_video_frames` and every existing module.
- The compression is applied in **linear light**, per channel, with the same affine on all three channels. Never in sRGB space.
- Strength `s` is always in `[0.0, 1.0]`. `s=0` is identity; `s=1` makes the frame a single colour.
- Delta measurement must use motion compensation (`estimate_global_shift` + `compensate_shift`), matching `detector.pipeline._iter_scores_and_masks`.
- Pass/fail is decided **only** by `detector.pipeline.run_detection` over the **whole clip**, never by the analytic estimate and never on an isolated segment.
- The trailing reference window is `round(fps)` frames — the same length as `detector.scoring.WindowedFlashCounter`'s window.
- The segment margin is `round(0.5 * fps)` frames — the same value `detector.pipeline.run_detection` passes to `scores_to_segments`.
- Run tests from the repo root: `python -m pytest`. `pytest.ini` sets `pythonpath = .`.
- Existing tests must keep passing. The suite is 249 tests as of commit `1e28a6b`.

---

### Task 1: Public sRGB ↔ linear-light conversion

`fallback/transfer.py` needs to go to linear light and back. `detector/luminance.py` already has the forward conversion as a private helper and has no inverse.

**Files:**
- Modify: `detector/luminance.py` (whole file — it is 20 lines)
- Test: `detector/tests/test_luminance.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `srgb_to_linear(channel: np.ndarray) -> np.ndarray` and `linear_to_srgb(channel: np.ndarray) -> np.ndarray`, exact inverses of each other over `[0, 1]`.

- [ ] **Step 1: Write the failing tests**

Append to `detector/tests/test_luminance.py`:

```python
def test_srgb_to_linear_round_trips_through_linear_to_srgb():
    from detector.luminance import linear_to_srgb, srgb_to_linear

    values = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    assert np.allclose(linear_to_srgb(srgb_to_linear(values)), values, atol=1e-6)


def test_linear_to_srgb_is_finite_for_negative_input():
    # Compression clamps its output before converting back, but a fractional
    # exponent on a negative base yields NaN, and a silent NaN frame is far
    # worse than a clamped one -- the conversion must not be the thing that
    # produces it.
    from detector.luminance import linear_to_srgb

    assert np.isfinite(linear_to_srgb(np.array([-0.5, 0.0, 0.5], dtype=np.float32))).all()


def test_srgb_linear_boundary_values_are_exact():
    from detector.luminance import linear_to_srgb, srgb_to_linear

    assert srgb_to_linear(np.array([0.0], dtype=np.float32))[0] == pytest.approx(0.0)
    assert srgb_to_linear(np.array([1.0], dtype=np.float32))[0] == pytest.approx(1.0)
    assert linear_to_srgb(np.array([0.0], dtype=np.float32))[0] == pytest.approx(0.0)
    assert linear_to_srgb(np.array([1.0], dtype=np.float32))[0] == pytest.approx(1.0)
```

If `detector/tests/test_luminance.py` does not already import `pytest` and `numpy as np`, add those imports at the top.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest detector/tests/test_luminance.py -v`
Expected: FAIL — `ImportError: cannot import name 'srgb_to_linear'`

- [ ] **Step 3: Implement**

Replace the body of `detector/luminance.py` with:

```python
"""Relative luminance per WCAG/BT.709: sRGB -> linear -> weighted sum, plus
the sRGB transfer function in both directions.

The conversions are public because working in linear light is not a
detector-internal concern: any correction that scales or offsets pixel
values has to do it on linear values, or the sRGB curve turns a uniform
scale into a per-channel colour shift.
"""
import numpy as np

_SRGB_A = 0.055
_SRGB_LINEAR_CUTOFF = 0.0031308
_SRGB_ENCODED_CUTOFF = 0.04045
_R_COEFF, _G_COEFF, _B_COEFF = 0.2126, 0.7152, 0.0722


def srgb_to_linear(channel: np.ndarray) -> np.ndarray:
    low = channel <= _SRGB_ENCODED_CUTOFF
    # np.where evaluates both branches, so the power branch sees every
    # element including the ones the mask discards -- clamp its base at zero
    # so a negative input cannot produce NaN in the unselected half.
    safe = np.clip(channel, 0.0, None)
    return np.where(low, channel / 12.92, ((safe + _SRGB_A) / (1 + _SRGB_A)) ** 2.4)


def linear_to_srgb(channel: np.ndarray) -> np.ndarray:
    low = channel <= _SRGB_LINEAR_CUTOFF
    safe = np.clip(channel, 0.0, None)
    return np.where(low, channel * 12.92, (1 + _SRGB_A) * safe ** (1 / 2.4) - _SRGB_A)


def relative_luminance(frame_rgb: np.ndarray) -> np.ndarray:
    linear = srgb_to_linear(frame_rgb.astype(np.float32))
    r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]
    return _R_COEFF * r + _G_COEFF * g + _B_COEFF * b
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS, 252 tests (249 existing + 3 new). `relative_luminance` behaviour is unchanged, so every existing luminance/detector test must still pass — if any fails, the refactor changed behaviour and must be fixed, not the test.

- [ ] **Step 5: Commit**

```bash
git add detector/luminance.py detector/tests/test_luminance.py
git commit -m "feat(detector): expose sRGB <-> linear conversion in both directions"
```

---

### Task 2: Contrast compression transfer function

**Files:**
- Create: `fallback/__init__.py` (empty)
- Create: `fallback/transfer.py`
- Create: `fallback/tests/__init__.py` (empty)
- Test: `fallback/tests/test_transfer.py`

**Interfaces:**
- Consumes: `detector.luminance.srgb_to_linear`, `detector.luminance.linear_to_srgb`.
- Produces:
  - `trailing_reference(frames_linear: list[np.ndarray], index: int, window_frames: int) -> np.ndarray` — per-channel mean over the trailing window ending at (and including) `index`; returns shape `(3,)` float32.
  - `compress_contrast(frame_rgb: np.ndarray, reference_linear: np.ndarray, strength: float) -> np.ndarray` — sRGB in, sRGB out.

- [ ] **Step 1: Write the failing tests**

Create `fallback/tests/test_transfer.py`:

```python
import numpy as np
import pytest

from detector.luminance import linear_to_srgb, relative_luminance, srgb_to_linear
from fallback.transfer import compress_contrast, trailing_reference


def _texture(h=16, w=16, seed=0):
    return (np.random.default_rng(seed).random((h, w, 3)).astype(np.float32) * 0.8 + 0.1)


def test_strength_zero_is_the_identity():
    frame = _texture()
    reference = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    assert np.allclose(compress_contrast(frame, reference, 0.0), frame, atol=1e-5)


def test_strength_one_makes_every_pixel_the_reference_colour():
    frame = _texture()
    reference = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    out = compress_contrast(frame, reference, 1.0)
    expected = linear_to_srgb(reference)
    assert np.allclose(out, expected[None, None, :], atol=1e-5)


def test_luminance_delta_scales_by_one_minus_strength():
    # This is the property the analytic strength estimate in strength.py
    # rests on: if it does not hold, the computed strength is meaningless.
    dark = _texture(seed=1) * 0.3
    bright = np.clip(dark + 0.4, 0.0, 1.0)
    reference = np.array([0.25, 0.25, 0.25], dtype=np.float32)

    before = np.abs(relative_luminance(bright) - relative_luminance(dark))
    for strength in (0.25, 0.5, 0.75):
        after = np.abs(
            relative_luminance(compress_contrast(bright, reference, strength))
            - relative_luminance(compress_contrast(dark, reference, strength))
        )
        assert np.allclose(after, before * (1.0 - strength), atol=1e-5)


def test_output_stays_in_range():
    frame = _texture()
    reference = np.array([0.9, 0.9, 0.9], dtype=np.float32)
    out = compress_contrast(frame, reference, 0.5)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_grey_input_with_grey_reference_stays_grey():
    # Compressing in sRGB space instead of linear light would pull the three
    # channels by different amounts and tint a neutral frame.
    frame = np.full((8, 8, 3), 0.7, dtype=np.float32)
    frame[:4] = 0.2
    reference = np.array([0.4, 0.4, 0.4], dtype=np.float32)
    out = compress_contrast(frame, reference, 0.6)
    assert np.allclose(out[..., 0], out[..., 1], atol=1e-6)
    assert np.allclose(out[..., 1], out[..., 2], atol=1e-6)


def test_strength_outside_the_unit_interval_is_rejected():
    frame = _texture()
    reference = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    with pytest.raises(ValueError):
        compress_contrast(frame, reference, 1.5)
    with pytest.raises(ValueError):
        compress_contrast(frame, reference, -0.1)


def test_trailing_reference_averages_the_window_ending_at_index():
    frames_linear = [np.full((4, 4, 3), value, dtype=np.float32) for value in (0.1, 0.2, 0.3, 0.9)]
    assert np.allclose(trailing_reference(frames_linear, 2, 3), [0.2, 0.2, 0.2], atol=1e-6)


def test_trailing_reference_clamps_at_the_clip_start():
    # Frame 0 has no history; averaging a window that runs off the front must
    # use what exists rather than index negatively and silently wrap around
    # to the end of the clip.
    frames_linear = [np.full((4, 4, 3), value, dtype=np.float32) for value in (0.1, 0.5, 0.9)]
    assert np.allclose(trailing_reference(frames_linear, 0, 5), [0.1, 0.1, 0.1], atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest fallback/tests/test_transfer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fallback'`

- [ ] **Step 3: Implement**

Create `fallback/__init__.py` and `fallback/tests/__init__.py` as empty files.

Create `fallback/transfer.py`:

```python
"""The Tier 0 correction itself: pull a frame's pixels toward a reference
colour, in linear light, by a strength in [0, 1].

Why compression and not mean-matching: scaling a frame so its *mean*
luminance matches a reference leaves the ratios inside the frame untouched,
so the per-pixel differences the Detector actually thresholds on survive
almost unchanged -- brightening the dark frames by as much as it darkens
the bright ones. An affine pull toward a single colour is what collapses
those per-pixel differences, which is why Apple's published Video Flashing
Reduction transfer function has the same affine shape.

Why linear light: the sRGB curve is non-linear, so an affine map applied to
encoded values scales each channel differently and tints the picture. In
linear light the same affine on all three channels gives exactly the same
affine on relative luminance, since luminance is a fixed weighted sum of
the linear channels. That equality is what lets strength.py compute the
required strength instead of searching for it.
"""
import numpy as np

from detector.luminance import linear_to_srgb, srgb_to_linear


def trailing_reference(
    frames_linear: list[np.ndarray], index: int, window_frames: int
) -> np.ndarray:
    """Per-channel mean over the window of `window_frames` frames ending at
    (and including) `index`, clamped at the start of the clip.

    Trailing rather than centred so the correction stays causal and can be
    lifted into a streaming path later without changing its output.
    """
    start = max(0, index - window_frames + 1)
    window = frames_linear[start : index + 1]
    return np.mean([frame.reshape(-1, 3).mean(axis=0) for frame in window], axis=0).astype(
        np.float32
    )


def compress_contrast(
    frame_rgb: np.ndarray, reference_linear: np.ndarray, strength: float
) -> np.ndarray:
    """Compress `frame_rgb` toward `reference_linear` by `strength`.

    `strength` 0 returns the frame unchanged; 1 returns a frame of the
    reference colour, which has no internal contrast and therefore no
    per-pixel difference from the next such frame.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")

    linear = srgb_to_linear(frame_rgb.astype(np.float32))
    compressed = reference_linear + (linear - reference_linear) * (1.0 - strength)
    return linear_to_srgb(np.clip(compressed, 0.0, 1.0)).astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest fallback/tests/test_transfer.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add fallback/__init__.py fallback/transfer.py fallback/tests/__init__.py fallback/tests/test_transfer.py
git commit -m "feat(fallback): contrast compression transfer function"
```

---

### Task 3: Analytic strength estimate

**Files:**
- Create: `fallback/strength.py`
- Test: `fallback/tests/test_strength.py`

**Interfaces:**
- Consumes: `detector.luminance.relative_luminance`, `detector.motion.estimate_global_shift`, `detector.motion.compensate_shift`, `detector.profiles.ThresholdProfile`.
- Produces:
  - `motion_compensated_deltas(frames: list[np.ndarray]) -> list[np.ndarray]` — `len(frames) - 1` arrays of per-pixel `|Δ luminance|`.
  - `required_strength(frames: list[np.ndarray], profile: ThresholdProfile) -> float`.

- [ ] **Step 1: Write the failing tests**

Create `fallback/tests/test_strength.py`:

```python
import numpy as np
import pytest

from detector.luminance import relative_luminance
from detector.profiles import ThresholdProfile
from fallback.strength import motion_compensated_deltas, required_strength

PROFILE = ThresholdProfile(
    name="test", max_flashes_per_second=3, max_area_ratio=0.25,
    general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
    red_saturation_ratio_threshold=0.80,
)


def _texture(h=32, w=32, seed=0, scale=0.3):
    return (np.random.default_rng(seed).random((h, w, 3)).astype(np.float32) * scale)


def test_required_strength_matches_the_quantile_formula():
    # Half the pixels get a large brightness step, half stay put. With
    # max_area_ratio 0.25 the binding quantile is the 75th percentile of the
    # per-pixel delta, and the strength that brings it to the 0.10 threshold
    # is 1 - 0.10/q75. Computed here independently of the implementation.
    dark = _texture(seed=1)
    bright = dark.copy()
    bright[:, :16] = np.clip(bright[:, :16] + 0.6, 0.0, 1.0)

    delta = np.abs(relative_luminance(bright) - relative_luminance(dark))
    q75 = float(np.quantile(delta, 0.75))
    expected = 1.0 - 0.10 / q75

    assert required_strength([dark, bright], PROFILE) == pytest.approx(expected, abs=5e-3)


def test_required_strength_is_zero_for_an_already_safe_clip():
    # A static clip has no transitions at all, so no compression is owed.
    frame = _texture(seed=2)
    assert required_strength([frame, frame, frame], PROFILE) == 0.0


def test_required_strength_takes_the_worst_frame_pair():
    # One severe pair among mild ones must set the strength: if any pair
    # trips the area criterion the whole segment is flagged.
    base = _texture(seed=3)
    mild = np.clip(base + 0.02, 0.0, 1.0)
    severe = np.clip(base + 0.7, 0.0, 1.0)

    mild_only = required_strength([base, mild], PROFILE)
    with_severe = required_strength([base, mild, severe], PROFILE)
    assert with_severe > mild_only


def test_required_strength_never_exceeds_one():
    black = np.zeros((16, 16, 3), dtype=np.float32)
    white = np.ones((16, 16, 3), dtype=np.float32)
    assert 0.0 <= required_strength([black, white], PROFILE) <= 1.0


def test_required_strength_of_a_single_frame_is_zero():
    # No pair, no measurable transition -- must not raise on an empty
    # delta list.
    assert required_strength([_texture()], PROFILE) == 0.0


def test_motion_compensation_suppresses_deltas_from_a_pure_pan():
    # A translated copy of the same texture is not a flash. Without
    # compensation every edge pixel would register a large delta and the
    # estimate would demand needless compression on any panning shot.
    frame = _texture(h=64, w=64, seed=4, scale=0.9)
    shifted = np.roll(frame, shift=3, axis=1)

    compensated = motion_compensated_deltas([frame, shifted])[0]
    naive = np.abs(relative_luminance(shifted) - relative_luminance(frame))
    assert compensated.mean() < naive.mean() * 0.5


def test_motion_compensated_deltas_length_is_one_less_than_frames():
    frames = [_texture(seed=i) for i in range(4)]
    assert len(motion_compensated_deltas(frames)) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest fallback/tests/test_strength.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fallback.strength'`

- [ ] **Step 3: Implement**

Create `fallback/strength.py`:

```python
"""How much compression this footage actually needs, computed rather than
searched.

`transfer.compress_contrast` scales the per-pixel luminance difference
between consecutive frames by exactly (1 - strength) as long as the
reference barely moves between them, which a one-second trailing mean does.
`detector.flash.transition_mask` flags a pixel when that difference exceeds
`general_flash_delta_threshold`, and `detector.segments._is_risky` fires
when the flagged fraction exceeds `max_area_ratio`. So the binding quantity
is the delta at the (1 - max_area_ratio) quantile: bring that to the
threshold and the flagged area lands at the limit.

The result is an estimate, not a verdict. Three things it cannot see:
`transition_mask` also requires the darker of the two frames to be below
`general_flash_dark_threshold`, and compression moves pixels across that
line in both directions; `flash.red_flash_mask` is a boolean XOR of
"is this pixel saturated red", which converges to zero as strength
approaches 1 but not proportionally to (1 - strength); and the flagged area
the profile is compared against is a maximum over a one-second window, not
a per-pair quantity. `apply.mitigate_with_fallback` therefore treats this
as a starting point and lets the real Detector decide.
"""
import numpy as np

from detector.luminance import relative_luminance
from detector.motion import compensate_shift, estimate_global_shift
from detector.profiles import ThresholdProfile


def motion_compensated_deltas(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Per-pixel |luminance change| between consecutive frames, after
    cancelling global camera motion.

    Mirrors `detector.pipeline._iter_scores_and_masks`: it compares
    `compensate_shift(prev)` against `curr`, so measuring without
    compensation would count a pan's edge displacement as a flash and
    demand compression that the Detector never asked for.
    """
    luminances = [relative_luminance(frame) for frame in frames]
    deltas = []
    for previous, current in zip(luminances[:-1], luminances[1:]):
        dx, dy = estimate_global_shift(previous, current)
        aligned_previous = compensate_shift(previous, dx, dy)
        deltas.append(np.abs(current - aligned_previous))
    return deltas


def required_strength(frames: list[np.ndarray], profile: ThresholdProfile) -> float:
    """Smallest strength that brings every frame pair's flagged area to
    `profile.max_area_ratio`, or 0.0 if none of them exceed it."""
    threshold = profile.general_flash_delta_threshold
    quantile = 1.0 - profile.max_area_ratio

    strongest = 0.0
    for delta in motion_compensated_deltas(frames):
        binding_delta = float(np.quantile(delta, quantile))
        if binding_delta > threshold:
            strongest = max(strongest, 1.0 - threshold / binding_delta)
    return float(min(1.0, strongest))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest fallback/tests/test_strength.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add fallback/strength.py fallback/tests/test_strength.py
git commit -m "feat(fallback): compute the required compression strength analytically"
```

---

### Task 4: Segment application, fade, and the verify/escalate loop

This is the task that produces the deliverable the whole plan exists for.

**Files:**
- Create: `fallback/apply.py`
- Test: `fallback/tests/test_apply.py`

**Interfaces:**
- Consumes: `fallback.transfer.trailing_reference`, `fallback.transfer.compress_contrast`, `fallback.strength.required_strength`, `detector.pipeline.run_detection`, `detector.luminance.srgb_to_linear`, `detector.segments.RiskSegment`.
- Produces:
  - `fade_weights(length: int, margin_frames: int) -> np.ndarray` — float32, length `length`, values in `[0, 1]`.
  - `SegmentOutcome` dataclass with fields `start_frame: int`, `end_frame: int`, `initial_strength: float`, `final_strength: float`, `rounds: int`, `passed: bool`.
  - `FallbackReport` dataclass with fields `segments: list[SegmentOutcome]`, `passed: bool`, `rounds: int`, `remaining_segments: int`.
  - `mitigate_with_fallback(frames, fps, profile, max_rounds=6, strength_step=0.1) -> tuple[list[np.ndarray], FallbackReport]`.

- [ ] **Step 1: Write the failing tests for the fade**

Create `fallback/tests/test_apply.py`:

```python
import numpy as np
import pytest

from detector.pipeline import run_detection
from detector.profiles import ThresholdProfile
from fallback.apply import FallbackReport, fade_weights, mitigate_with_fallback

PROFILE = ThresholdProfile(
    name="test", max_flashes_per_second=3, max_area_ratio=0.25,
    general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
    red_saturation_ratio_threshold=0.80,
)


def test_fade_starts_and_ends_near_zero_and_is_full_in_the_middle():
    weights = fade_weights(length=40, margin_frames=10)
    assert weights[0] < 0.1
    assert weights[-1] < 0.1
    assert weights[20] == pytest.approx(1.0)


def test_fade_is_monotonic_on_the_way_in_and_out():
    weights = fade_weights(length=40, margin_frames=10)
    assert np.all(np.diff(weights[:10]) > 0)
    assert np.all(np.diff(weights[-10:]) < 0)


def test_fade_never_exceeds_one_or_drops_below_zero():
    for length, margin in ((40, 10), (5, 10), (1, 10), (2, 1)):
        weights = fade_weights(length, margin)
        assert weights.min() >= 0.0 and weights.max() <= 1.0
        assert len(weights) == length


def test_fade_halves_itself_when_the_segment_is_shorter_than_two_margins():
    # A 6-frame segment with a 10-frame margin would otherwise write two
    # overlapping 10-frame ramps into a 6-slot array, and the second would
    # overwrite the first -- producing a step exactly where the fade exists
    # to prevent one.
    weights = fade_weights(length=6, margin_frames=10)
    assert np.all(np.diff(weights[:3]) > 0)
    assert np.all(np.diff(weights[3:]) < 0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest fallback/tests/test_apply.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fallback.apply'`

- [ ] **Step 3: Implement `fade_weights` and the dataclasses**

Create `fallback/apply.py`:

```python
"""Tier 0 end to end: pick a strength per risk segment, apply it with a
fade, ask the Detector, raise the strength on whatever still fails.

Only risk segments are touched. Compressing the whole clip would be simpler
and is what the reference blind-deflickering implementation does -- and it
costs 21-35% of the mean luminance across the entire video, including the
parts that were never dangerous. Leaving safe frames untouched is the one
place this pipeline can be better than that, so it is worth the fade
machinery needed to re-enter a segment without a visible step.
"""
from dataclasses import dataclass

import numpy as np

from detector.luminance import srgb_to_linear
from detector.pipeline import run_detection
from detector.profiles import ThresholdProfile
from detector.segments import RiskSegment
from fallback.strength import required_strength
from fallback.transfer import compress_contrast, trailing_reference


@dataclass
class SegmentOutcome:
    start_frame: int
    end_frame: int
    initial_strength: float
    final_strength: float
    rounds: int
    passed: bool


@dataclass
class FallbackReport:
    segments: list[SegmentOutcome]
    passed: bool
    rounds: int
    remaining_segments: int


def fade_weights(length: int, margin_frames: int) -> np.ndarray:
    """Raised-cosine ramp in, flat, ramp out -- one multiplier per frame.

    Without it the strength steps from 0 to its full value at the segment
    boundary, and that step is itself a luminance transition the Detector
    can flag -- the correction would create the hazard it exists to remove.
    The ramp lives in the +-0.5s margin `run_detection` already pads each
    segment with, which is outside the risky frames proper.
    """
    weights = np.ones(length, dtype=np.float32)
    fade = min(margin_frames, length // 2)
    if fade <= 0:
        return weights

    steps = (np.arange(fade, dtype=np.float32) + 1.0) / (fade + 1.0)
    ramp = (0.5 * (1.0 - np.cos(np.pi * steps))).astype(np.float32)
    weights[:fade] = ramp
    weights[length - fade :] = ramp[::-1]
    return weights
```

- [ ] **Step 4: Run the fade tests**

Run: `python -m pytest fallback/tests/test_apply.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add fallback/apply.py fallback/tests/test_apply.py
git commit -m "feat(fallback): raised-cosine fade over the segment margin"
```

- [ ] **Step 6: Write the failing tests for the escalation loop**

Append to `fallback/tests/test_apply.py`:

```python
def _flickering_clip(n_frames=40, h=48, w=48, seed=0, flicker_range=None):
    """A clip where a large region alternates bright/dark hard enough that
    run_detection flags it -- the input Tier 0 exists to fix.

    `flicker_range` limits the flashing to `[lo, hi)` so a test can have
    genuinely safe frames outside the risk segment; the default flickers
    throughout.
    """
    rng = np.random.default_rng(seed)
    base = rng.random((h, w, 3)).astype(np.float32) * 0.2
    low, high = flicker_range if flicker_range is not None else (0, n_frames)
    frames = []
    for i in range(n_frames):
        frame = base.copy()
        if low <= i < high and i % 2 == 0:
            frame[:, : int(w * 0.6)] = 0.95
        frames.append(frame)
    return frames


def test_the_test_clip_is_actually_risky_before_mitigation():
    # Guards the two tests below: if the fixture stopped being risky they
    # would pass trivially without exercising anything.
    _scores, segments = run_detection(_flickering_clip(), fps=20.0, profile=PROFILE)
    assert len(segments) > 0


def test_mitigate_with_fallback_clears_every_risk_segment():
    frames = _flickering_clip()
    corrected, report = mitigate_with_fallback(frames, fps=20.0, profile=PROFILE)

    _scores, segments = run_detection(corrected, fps=20.0, profile=PROFILE)
    assert segments == []
    assert report.passed is True
    assert report.remaining_segments == 0


def test_a_safe_clip_is_returned_untouched():
    rng = np.random.default_rng(7)
    frames = [rng.random((32, 32, 3)).astype(np.float32) * 0.1 for _ in range(30)]
    corrected, report = mitigate_with_fallback(frames, fps=20.0, profile=PROFILE)

    assert report.segments == []
    assert report.passed is True
    for original, result in zip(frames, corrected):
        assert np.array_equal(original, result)


def test_report_records_the_strength_actually_used():
    frames = _flickering_clip()
    _corrected, report = mitigate_with_fallback(frames, fps=20.0, profile=PROFILE)

    assert len(report.segments) > 0
    for outcome in report.segments:
        assert 0.0 < outcome.final_strength <= 1.0
        assert outcome.final_strength >= outcome.initial_strength
        assert outcome.rounds >= 0


def test_frames_outside_every_segment_are_bit_identical():
    # The mask-free global correction still must not touch frames the
    # Detector called safe -- that is the whole reason for working per
    # segment instead of over the whole clip.
    #
    # The flicker is confined to the middle so safe frames actually exist;
    # a clip that flickers throughout would put every frame inside a
    # segment and let this pass without checking anything.
    frames = _flickering_clip(n_frames=100, flicker_range=(40, 60))
    corrected, report = mitigate_with_fallback(frames, fps=20.0, profile=PROFILE)

    inside = set()
    for outcome in report.segments:
        inside.update(range(outcome.start_frame, outcome.end_frame + 1))
    outside = [i for i in range(len(frames)) if i not in inside]

    assert outside, "no frames outside a segment -- the assertion below would be vacuous"
    for index in outside:
        assert np.array_equal(frames[index], corrected[index])


def test_max_rounds_caps_the_escalation_loop():
    # The loop must stop at max_rounds and return a report rather than
    # raising or spinning -- the caller decides the policy for a clip that
    # did not converge (README section 7 logs the incident).
    frames = _flickering_clip()
    _corrected, report = mitigate_with_fallback(
        frames, fps=20.0, profile=PROFILE, max_rounds=1, strength_step=0.0
    )
    assert isinstance(report, FallbackReport)
    assert report.rounds <= 1
```

- [ ] **Step 7: Run to verify the new tests fail**

Run: `python -m pytest fallback/tests/test_apply.py -v`
Expected: FAIL — `ImportError: cannot import name 'mitigate_with_fallback'`

- [ ] **Step 8: Implement the application and escalation loop**

Append to `fallback/apply.py`:

```python
def _apply_segments(
    frames: list[np.ndarray],
    frames_linear: list[np.ndarray],
    segments: list[RiskSegment],
    strengths: dict[tuple[int, int], float],
    window_frames: int,
    margin_frames: int,
) -> list[np.ndarray]:
    """Rebuild the clip with each segment compressed at its own strength.

    Frames outside every segment are passed through by identity (the same
    array object), not recomputed at strength 0 -- an sRGB round trip is
    lossy at float32 and would leave a measurable difference on frames the
    Detector already called safe.
    """
    output = list(frames)
    for segment in segments:
        length = segment.end_frame - segment.start_frame + 1
        weights = fade_weights(length, margin_frames)
        strength = strengths[(segment.start_frame, segment.end_frame)]
        for offset in range(length):
            index = segment.start_frame + offset
            reference = trailing_reference(frames_linear, index, window_frames)
            output[index] = compress_contrast(
                frames[index], reference, float(strength * weights[offset])
            )
    return output


def mitigate_with_fallback(
    frames: list[np.ndarray],
    fps: float,
    profile: ThresholdProfile,
    max_rounds: int = 6,
    strength_step: float = 0.1,
) -> tuple[list[np.ndarray], FallbackReport]:
    """Compress every risk segment until the Detector passes the whole clip.

    Verification runs over the whole clip every round, never over a segment
    in isolation: `WindowedFlashCounter` primes its window with flagged
    slots and marks frame 0 uncertain, so a segment measured on its own is
    judged under different conditions than the one that decides pass or
    fail.
    """
    _scores, segments = run_detection(frames, fps=fps, profile=profile)
    if not segments:
        return list(frames), FallbackReport(
            segments=[], passed=True, rounds=0, remaining_segments=0
        )

    window_frames = max(1, round(fps))
    margin_frames = round(0.5 * fps)

    keys = [(segment.start_frame, segment.end_frame) for segment in segments]
    initial = {
        key: required_strength(frames[key[0] : key[1] + 1], profile) for key in keys
    }
    # A segment the analytic estimate rates as needing nothing still has to
    # move: run_detection flagged it, so something the estimate cannot see
    # (the red-flash XOR, the dark-threshold condition, the windowed area
    # maximum) is driving it. Starting at exactly 0 would waste a round.
    strengths = {key: max(value, strength_step) for key, value in initial.items()}
    rounds_used = {key: 0 for key in keys}

    frames_linear = [srgb_to_linear(frame.astype(np.float32)) for frame in frames]

    output = list(frames)
    remaining: list[RiskSegment] = []
    passed = False
    rounds = 0

    for rounds in range(1, max_rounds + 1):
        output = _apply_segments(
            frames, frames_linear, segments, strengths, window_frames, margin_frames
        )
        _scores, remaining = run_detection(output, fps=fps, profile=profile)
        if not remaining:
            passed = True
            break

        risky_frames = set()
        for segment in remaining:
            risky_frames.update(range(segment.start_frame, segment.end_frame + 1))

        for segment, key in zip(segments, keys):
            overlaps = risky_frames.intersection(
                range(segment.start_frame, segment.end_frame + 1)
            )
            if overlaps:
                strengths[key] = min(1.0, strengths[key] + strength_step)
                rounds_used[key] += 1

    still_risky = set()
    for segment in remaining:
        still_risky.update(range(segment.start_frame, segment.end_frame + 1))

    outcomes = [
        SegmentOutcome(
            start_frame=key[0],
            end_frame=key[1],
            initial_strength=initial[key],
            final_strength=strengths[key],
            rounds=rounds_used[key],
            passed=not still_risky.intersection(range(key[0], key[1] + 1)),
        )
        for key in keys
    ]

    return output, FallbackReport(
        segments=outcomes,
        passed=passed,
        rounds=rounds,
        remaining_segments=len(remaining),
    )
```

- [ ] **Step 9: Run the whole file**

Run: `python -m pytest fallback/tests/test_apply.py -v`
Expected: PASS, 10 tests.

If `test_mitigate_with_fallback_clears_every_risk_segment` fails, do **not** weaken the assertion. Print `report.segments` and read the strengths: if they reached 1.0 and detection still fails, the clip is uniform-colour and still flagged, which would mean a real bug in the transfer function (a uniform frame has zero per-pixel delta and cannot be flagged by `transition_mask`).

- [ ] **Step 10: Run the full suite and commit**

Run: `python -m pytest -q`
Expected: PASS. The suite is now 277 tests (249 baseline + 3 from Task 1 + 8 from Task 2 + 7 from Task 3 + 10 here).

```bash
git add fallback/apply.py fallback/tests/test_apply.py
git commit -m "feat(fallback): per-segment application with verify/escalate loop"
```

---

### Task 5: Video-in / video-out CLI

**Files:**
- Create: `fallback/cli.py`
- Test: `fallback/tests/test_cli.py`

**Interfaces:**
- Consumes: `fallback.apply.mitigate_with_fallback`, `detector.cli.read_video_frames`, `mitigator.cli.write_video`, `detector.profiles.load_profile`.
- Produces: `main(argv: list[str] | None = None) -> int` — exit code 0 when the clip passes, 1 when it does not.

- [ ] **Step 1: Write the failing test**

Create `fallback/tests/test_cli.py`:

```python
import json

import cv2
import numpy as np

from fallback.cli import main


def _write_flickering_video(path, n_frames=40, h=48, w=48, fps=20.0):
    rng = np.random.default_rng(0)
    base = (rng.random((h, w, 3)) * 0.2 * 255).astype(np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for i in range(n_frames):
            frame = base.copy()
            if i % 2 == 0:
                frame[:, : int(w * 0.6)] = 242
            writer.write(frame)
    finally:
        writer.release()


def test_cli_writes_a_video_and_a_report(tmp_path):
    source = tmp_path / "in.mp4"
    _write_flickering_video(source)
    output = tmp_path / "out.mp4"
    report = tmp_path / "report.json"

    exit_code = main([
        "--input", str(source),
        "--output", str(output),
        "--profile", "configs/profiles/itu.json",
        "--report", str(report),
    ])

    assert exit_code == 0
    assert output.exists()

    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["passed"] is True
    assert "segments" in data

    written = cv2.VideoCapture(str(output))
    try:
        assert int(written.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    finally:
        written.release()


def test_cli_returns_nonzero_when_the_clip_still_fails(tmp_path, monkeypatch):
    # The exit code is how a caller learns mitigation did not converge, so
    # it is asserted against a report that says so. Forcing a real clip to
    # fail would make this test depend on how well the escalation loop
    # happens to do on one fixture, which is test_apply.py's subject, not
    # the CLI's contract.
    import fallback.cli as cli_module
    from fallback.apply import FallbackReport, SegmentOutcome

    source = tmp_path / "in.mp4"
    _write_flickering_video(source)

    def _fake_mitigate(frames, fps, profile, max_rounds, strength_step):
        outcome = SegmentOutcome(
            start_frame=0, end_frame=9, initial_strength=0.5,
            final_strength=1.0, rounds=6, passed=False,
        )
        return list(frames), FallbackReport(
            segments=[outcome], passed=False, rounds=6, remaining_segments=1
        )

    monkeypatch.setattr(cli_module, "mitigate_with_fallback", _fake_mitigate)

    report = tmp_path / "report.json"
    exit_code = main([
        "--input", str(source),
        "--output", str(tmp_path / "out.mp4"),
        "--profile", "configs/profiles/itu.json",
        "--report", str(report),
    ])
    assert exit_code == 1
    assert json.loads(report.read_text(encoding="utf-8"))["passed"] is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest fallback/tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fallback.cli'`

- [ ] **Step 3: Implement**

Create `fallback/cli.py`:

```python
"""Batch CLI for the Tier 0 fallback: video in, video out, plus a JSON
report of the strength each segment needed.

The report is the point as much as the video is. The strengths it records
are the first concrete measurement of how much contrast has to be crushed
to clear the profile's area limit on real footage -- the number the neural
path has to beat, and the design input for replacing PriorCalc's target.

Audio is not preserved, matching mitigator/cli.py.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from detector.cli import read_video_frames
from detector.profiles import load_profile
from fallback.apply import mitigate_with_fallback
from mitigator.cli import write_video


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Tier 0 classical fallback over a video file."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--report", help="write a JSON report of per-segment strengths here")
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--strength-step", type=float, default=0.1)
    args = parser.parse_args(argv)

    frame_iter, fps = read_video_frames(args.input)
    frames = list(frame_iter)
    profile = load_profile(Path(args.profile))

    corrected, report = mitigate_with_fallback(
        frames,
        fps=fps,
        profile=profile,
        max_rounds=args.max_rounds,
        strength_step=args.strength_step,
    )
    write_video(corrected, Path(args.output), fps)

    payload = {
        "input": args.input,
        "output": args.output,
        "profile": profile.name,
        "frames": len(frames),
        "fps": fps,
        "passed": report.passed,
        "rounds": report.rounds,
        "remaining_segments": report.remaining_segments,
        "segments": [asdict(outcome) for outcome in report.segments],
    }
    if args.report:
        Path(args.report).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"processed {len(frames)} frames -> {args.output}")
    for outcome in report.segments:
        print(
            f"  segment {outcome.start_frame}-{outcome.end_frame}: "
            f"strength {outcome.initial_strength:.3f} -> {outcome.final_strength:.3f} "
            f"({outcome.rounds} escalations, {'passed' if outcome.passed else 'FAILED'})"
        )
    if report.passed:
        print("Detector re-validation: PASSED (0 risk segments)")
        return 0

    # A clip that still fails is not a crash, but it must not look like
    # success either -- README section 7 treats an exhausted escalation as
    # an incident to log, not something to swallow.
    print(f"Detector re-validation: FAILED ({report.remaining_segments} segments remain)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest fallback/tests/test_cli.py -v`
Expected: PASS, 2 tests.

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest -q`
Expected: PASS, 279 tests.

```bash
git add fallback/cli.py fallback/tests/test_cli.py
git commit -m "feat(fallback): video-in/video-out CLI with a per-segment strength report"
```

---

### Task 6: Real-footage verification and the measurement record

The synthetic tests prove the machinery works. This task answers the questions the module was built to answer.

**Files:**
- Modify: `pytest.ini` (register the `slow` marker)
- Create: `fallback/tests/test_real_footage.py`
- Create: `docs/superpowers/specs/2026-08-07-tier0-fallback-measurements.md`

**Interfaces:**
- Consumes: `fallback.apply.mitigate_with_fallback`, `scripts.screen_clean_clips.read_frames_lenient`, `detector.pipeline.run_detection`.
- Produces: no code interface — a marked test and a measurement document.

- [ ] **Step 1: Register the marker**

Add to `pytest.ini` under `[pytest]`:

```ini
markers =
    slow: exercises real video files and takes minutes; excluded from the default run
addopts = -m "not slow"
```

- [ ] **Step 2: Write the real-footage test**

Create `fallback/tests/test_real_footage.py`:

```python
"""Real-clip verification. Excluded from the default run -- these decode
hundreds of full-resolution frames and re-run detection several times.

Run explicitly:  python -m pytest fallback/tests/test_real_footage.py -m slow -v -s
"""
from pathlib import Path

import numpy as np
import pytest

from detector.luminance import relative_luminance
from detector.pipeline import run_detection
from detector.profiles import load_profile
from fallback.apply import mitigate_with_fallback
from scripts.screen_clean_clips import read_frames_lenient

CLIP = Path("test_mp4/wonbon_640.mp4")


@pytest.mark.slow
@pytest.mark.skipif(not CLIP.exists(), reason=f"{CLIP} not present")
def test_real_concert_clip_passes_after_fallback():
    profile = load_profile(Path("configs/profiles/itu.json"))
    frames, fps, _note = read_frames_lenient(CLIP, None)

    _scores, before = run_detection(frames, fps=fps, profile=profile)
    assert before, "fixture clip is not risky -- the test would pass trivially"

    corrected, report = mitigate_with_fallback(frames, fps=fps, profile=profile)
    _scores, after = run_detection(corrected, fps=fps, profile=profile)

    def mean_luminance(clip):
        return float(np.mean([relative_luminance(f).mean() for f in clip]))

    def mean_contrast(clip):
        return float(np.mean([relative_luminance(f).std() for f in clip]))

    # Printed so the numbers land in the measurement document -- run with -s.
    print(f"\nsegments      {len(before)} -> {len(after)}")
    print(f"passed        {report.passed}   rounds {report.rounds}")
    for outcome in report.segments:
        print(
            f"  {outcome.start_frame}-{outcome.end_frame}: "
            f"{outcome.initial_strength:.3f} -> {outcome.final_strength:.3f} "
            f"({outcome.rounds} escalations)"
        )
    print(f"mean luminance {mean_luminance(frames):.4f} -> {mean_luminance(corrected):.4f}")
    print(f"mean contrast  {mean_contrast(frames):.4f} -> {mean_contrast(corrected):.4f}")

    assert after == []
    assert report.passed is True
```

- [ ] **Step 3: Verify the default run still excludes it**

Run: `python -m pytest -q`
Expected: PASS, 279 tests selected and 1 deselected — the slow test must not run by default.

- [ ] **Step 4: Run the slow test and capture the output**

Run: `python -m pytest fallback/tests/test_real_footage.py -m slow -v -s`
Expected: PASS. Takes several minutes (931 frames, detection per escalation round).

Copy the printed block verbatim; the next step needs it.

- [ ] **Step 5: Run the CLI on all three real clips**

```bash
python -m fallback.cli --input test_mp4/wonbon_640.mp4 --output test_mp4/out/tier0_wonbon.mp4 --profile configs/profiles/itu.json --report test_mp4/out/tier0_wonbon.json
```

Repeat for the Anyma and Cera Khin clips in `test_mp4/`, writing to `tier0_anyma.*` and `tier0_cera.*`. These two are portrait and full resolution; if a run exhausts memory, downscale first the way `test_mp4/wonbon_640.mp4` was produced (long side 640) and note the resolution used.

- [ ] **Step 6: Write the measurement document**

Create `docs/superpowers/specs/2026-08-07-tier0-fallback-measurements.md` with the real numbers from Steps 4 and 5 — no placeholders. It must contain:

- A table with one row per clip: resolution used, segments before/after, max `final_strength`, mean luminance before/after with the percentage change, mean contrast before/after with the percentage change.
- A table with one row per segment: `initial_strength`, `final_strength`, escalation rounds. The gap between the two columns is the size of the analytic estimate's error, which section 2.3 of the design predicted would be non-zero.
- A one-paragraph comparison against All-In-One-Deflicker's measured cost on the same clips: mean luminance −21% (`원본`), −35% (Anyma), −35% (Cera Khin). State plainly whether Tier 0 costs more or less.
- A one-paragraph verdict on design section 6's threshold question: if any clip's maximum `final_strength` is at or above 0.9, say so explicitly and record that the detection profile itself now warrants review, because near-uniform output is what the bar is demanding.

- [ ] **Step 7: Commit**

```bash
git add pytest.ini fallback/tests/test_real_footage.py docs/superpowers/specs/2026-08-07-tier0-fallback-measurements.md
git commit -m "test(fallback): real-footage verification and the strength measurements"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| 2.1 변환식 (linear light, per channel) | Task 2 |
| 2.2 `(1−s)` 성질, 분위수 공식 | Task 2 (property test), Task 3 (formula) |
| 2.3 근사인 이유 → 재검증 필수 | Task 4 (whole-clip loop), documented in `strength.py` |
| 2.4 페이드, 짧은 구간 처리 | Task 4 Steps 1–5 |
| 3.1 공개 인터페이스 | Tasks 2–4 (all signatures match) |
| 3.2 `detector/luminance.py` 변경 | Task 1 |
| 4 재검증/상향 루프, `max_rounds` 소진 | Task 4 Steps 6–10, Task 5 (exit code) |
| 5 테스트 전략 (전 항목) | Tasks 1–6 |
| 6 측정 산출물 | Task 6 |
| 7 제외 항목 | Not implemented, as specified |

**Type consistency:** `required_strength(frames, profile)` is used with exactly that signature in Task 4. `fade_weights(length, margin_frames)` matches between Task 4 Step 3 and its tests. `SegmentOutcome`/`FallbackReport` field names match between Task 4, Task 5 (`asdict`), and Task 6.

**One deviation from the spec, deliberate:** Task 4 seeds each segment's strength at `max(analytic_estimate, strength_step)` rather than the raw estimate. The Detector flagged the segment, so a zero estimate means the estimate cannot see what is driving it (spec section 2.3 lists three such causes) — starting at exactly zero would spend a full escalation round re-applying the identity. `initial_strength` in the report still records the raw estimate, so the measurement in Task 6 is unaffected.
