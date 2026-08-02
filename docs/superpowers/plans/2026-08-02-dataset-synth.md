# DatasetSynth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the offline DatasetSynth pipeline that injects PSE flash/red-flash patterns into clean clips, re-validates each synthesized sample with the real Detector before accepting it, and writes accepted (clean, degraded) frame pairs with metadata for later Mitigator training.

**Architecture:** Detector's own signal vocabulary (dark threshold, delta threshold, area ratio, red-saturation ratio, windowed flash-frequency) is reused as the synthesis parameter space, so accept/reject is a confirmation check rather than blind trial and error. Five stages: sample parameters per-profile (`training/params.py`) → inject pixels (`training/injection.py`) → synthesize-and-validate with retry (`training/synth.py`) → write to disk with resume support (`training/dataset_writer.py`) → batch CLI over clips × profiles × patterns (`training/cli.py`).

**Tech Stack:** Python 3.10+, numpy, opencv-python (`cv2`), pytest. No new dependencies beyond what `detector/` already uses.

## Global Constraints

- Synthesis parameters are computed **per individual `ThresholdProfile`** (one of the 6 shipped profiles), never from a single global min/max range spanning all profiles (per spec §2/§4).
- No new dependencies — only numpy/opencv-python/pytest/stdlib, matching `detector/`'s footprint (spec §1).
- A sample is written to disk **only** after `detector.pipeline.run_detection` on the synthesized clip confirms the injected window is risky under the target profile; there is exactly one accept path (spec §6 "구조적 안전장치").
- Frames outside the injected window, and pixels outside the injected mask, must remain byte-identical to the source clip — this is what keeps clean/degraded frame-aligned without extra bookkeeping (spec §5).
- Validation failure retries up to 5 times with freshly sampled parameters, then the (clip, profile, pattern, index) combination is skipped and recorded, never silently dropped (spec §6).
- Corrupt/unreadable source clips are skipped with a logged reason; the batch continues (spec §6) — unlike Detector's CLI, which fails the whole run on one bad video.
- Re-running the batch over an existing output directory skips samples that already exist there (spec §6, resume support).
- The automated test suite uses only small in-memory synthetic frames — no real DAVIS data required to run `pytest` (spec §7/§8).

---

## File Structure

```
flicker-guard/
├── training/
│   ├── __init__.py
│   ├── params.py          # Task 1
│   ├── injection.py       # Task 2
│   ├── synth.py            # Task 3
│   ├── dataset_writer.py  # Task 4
│   ├── cli.py              # Task 5
│   └── tests/
│       ├── __init__.py
│       ├── test_params.py
│       ├── test_injection.py
│       ├── test_synth.py
│       ├── test_dataset_writer.py
│       └── test_cli.py
```

---

## Task 1: Synthesis parameter sampling

**Files:**
- Create: `flicker-guard/training/params.py`
- Test: `flicker-guard/training/tests/test_params.py`

**Interfaces:**
- Consumes: `detector.profiles.ThresholdProfile`, `detector.profiles.load_profile` (for the shipped-profile test only).
- Produces: `InjectionWindow` dataclass (`start_frame: int`, `end_frame: int` inclusive, `mask_top: int`, `mask_left: int`, `mask_height: int`, `mask_width: int`, `period_frames: int`, `ramp_frames: int`), `SynthParams` dataclass (`pattern: str`, `window: InjectionWindow`, `dark_target: float | None`, `bright_target: float | None`, `red_rgb: tuple[float, float, float] | None`, `baseline_rgb: tuple[float, float, float] | None`), `sample_synthesis_params(profile: ThresholdProfile, pattern: str, clip_frame_count: int, frame_height: int, frame_width: int, fps: float, rng: np.random.Generator) -> SynthParams` (raises `ValueError` if the clip is too short).

`ramp_frames` records Detector's trailing-window length (`round(fps)`) at sampling time — Task 3's validator needs it to know how many frames at the start of the window are still "filling up" the windowed counter and can't yet show the steady-state flagged count.

- [ ] **Step 1: Write the failing test**

```python
# flicker-guard/training/tests/test_params.py
import pathlib

import numpy as np
import pytest

from detector.profiles import ThresholdProfile, load_profile
from training.params import sample_synthesis_params

PROFILE = ThresholdProfile(
    name="test", max_flashes_per_second=3, max_area_ratio=0.10,
    general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
    red_saturation_ratio_threshold=0.80,
)

SHIPPED_PROFILES_DIR = pathlib.Path(__file__).parent.parent.parent / "configs" / "profiles"


def test_sample_synthesis_params_general_flash_targets_exceed_profile_thresholds():
    rng = np.random.default_rng(0)
    params = sample_synthesis_params(PROFILE, "general", clip_frame_count=90, frame_height=100, frame_width=100, fps=30.0, rng=rng)
    assert params.pattern == "general"
    assert params.dark_target < PROFILE.general_flash_dark_threshold
    assert params.bright_target - params.dark_target > PROFILE.general_flash_delta_threshold
    assert params.bright_target <= 1.0


def test_sample_synthesis_params_red_flash_targets_exceed_ratio_threshold():
    rng = np.random.default_rng(0)
    params = sample_synthesis_params(PROFILE, "red", clip_frame_count=90, frame_height=100, frame_width=100, fps=30.0, rng=rng)
    assert params.pattern == "red"
    r, g, b = params.red_rgb
    assert r / (r + g + b) > PROFILE.red_saturation_ratio_threshold
    br, bg, bb = params.baseline_rgb
    assert br / (br + bg + bb) < PROFILE.red_saturation_ratio_threshold


def test_sample_synthesis_params_mask_area_exceeds_profile_ratio():
    rng = np.random.default_rng(0)
    params = sample_synthesis_params(PROFILE, "general", clip_frame_count=90, frame_height=100, frame_width=100, fps=30.0, rng=rng)
    w = params.window
    mask_area_ratio = (w.mask_height * w.mask_width) / (100 * 100)
    assert mask_area_ratio > PROFILE.max_area_ratio
    assert 0 <= w.mask_top and w.mask_top + w.mask_height <= 100
    assert 0 <= w.mask_left and w.mask_left + w.mask_width <= 100


def test_sample_synthesis_params_period_yields_enough_flagged_frames_per_window():
    rng = np.random.default_rng(0)
    fps = 30.0
    params = sample_synthesis_params(PROFILE, "general", clip_frame_count=90, frame_height=100, frame_width=100, fps=fps, rng=rng)
    window_frames = round(fps)
    flagged_per_window = window_frames // params.window.period_frames
    assert flagged_per_window > PROFILE.max_flashes_per_second


def test_sample_synthesis_params_window_fits_inside_clip_and_covers_full_second():
    rng = np.random.default_rng(0)
    fps = 30.0
    clip_frame_count = 90
    params = sample_synthesis_params(PROFILE, "general", clip_frame_count=clip_frame_count, frame_height=100, frame_width=100, fps=fps, rng=rng)
    w = params.window
    assert 1 <= w.start_frame
    assert w.end_frame < clip_frame_count
    assert (w.end_frame - w.start_frame + 1) >= round(fps)
    assert w.ramp_frames == round(fps)
    assert (w.end_frame - w.start_frame + 1) - w.ramp_frames >= 2 * w.period_frames


def test_sample_synthesis_params_raises_when_clip_too_short():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_synthesis_params(PROFILE, "general", clip_frame_count=5, frame_height=100, frame_width=100, fps=30.0, rng=rng)


@pytest.mark.parametrize("filename", ["kr.json", "jp.json", "itu.json", "ofcom.json", "w3c.json", "netflix.json"])
def test_sample_synthesis_params_exceeds_each_shipped_profile(filename):
    profile = load_profile(SHIPPED_PROFILES_DIR / filename)
    rng = np.random.default_rng(0)
    params = sample_synthesis_params(profile, "general", clip_frame_count=90, frame_height=100, frame_width=100, fps=30.0, rng=rng)
    mask_area_ratio = (params.window.mask_height * params.window.mask_width) / (100 * 100)
    assert mask_area_ratio > profile.max_area_ratio
    window_frames = round(30.0)
    flagged_per_window = window_frames // params.window.period_frames
    assert flagged_per_window > profile.max_flashes_per_second
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest flicker-guard/training/tests/test_params.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.params'`

- [ ] **Step 3: Write minimal implementation**

```python
# flicker-guard/training/params.py
"""Samples synthesis parameters guaranteed to exceed a ThresholdProfile's
thresholds with a comfortable margin, so Detector re-validation in
training.synth is a confirmation step rather than blind trial and error.

Per-profile, not global (spec Global Constraint): every value here is
computed from the specific ThresholdProfile passed in, never from a fixed
min/max spanning all profiles.

Frequency note: `ThresholdProfile.max_flashes_per_second` is compared
against Detector's `flagged_frame_count_last_second` (flagged *frames* in a
trailing one-second window), not the visual flash rate (see
detector/scoring.py). `period_frames` here is chosen so the number of
flagged frames inside any trailing one-second window exceeds
`max_flashes_per_second` with margin.
"""
from dataclasses import dataclass

import numpy as np

from detector.profiles import ThresholdProfile

_AREA_MARGIN = 0.10
_MAX_AREA_RATIO = 0.90
_RATIO_MARGIN = 0.05
_MAX_SATURATION_TARGET = 0.99


@dataclass
class InjectionWindow:
    start_frame: int
    end_frame: int  # inclusive
    mask_top: int
    mask_left: int
    mask_height: int
    mask_width: int
    period_frames: int  # frames per alternation half-cycle (state held this long)
    ramp_frames: int  # Detector's trailing-window length (round(fps)) at sample time


@dataclass
class SynthParams:
    pattern: str  # "general" | "red"
    window: InjectionWindow
    dark_target: float | None = None
    bright_target: float | None = None
    red_rgb: tuple[float, float, float] | None = None
    baseline_rgb: tuple[float, float, float] | None = None


def _mask_dims(frame_height: int, frame_width: int, target_area_ratio: float) -> tuple[int, int, int, int]:
    target_area_ratio = min(target_area_ratio, _MAX_AREA_RATIO)
    side_ratio = target_area_ratio ** 0.5
    mask_height = max(1, min(frame_height, round(frame_height * side_ratio)))
    mask_width = max(1, min(frame_width, round(frame_width * side_ratio)))
    mask_top = (frame_height - mask_height) // 2
    mask_left = (frame_width - mask_width) // 2
    return mask_top, mask_left, mask_height, mask_width


def _period_frames(profile: ThresholdProfile, fps: float) -> int:
    window_frames = max(1, round(fps))
    # Halve the just-crossing period for margin, so the steady-state flagged
    # count comfortably exceeds max_flashes_per_second rather than sitting
    # right at the boundary.
    period = window_frames / (2 * (profile.max_flashes_per_second + 1))
    return max(1, int(period))


def sample_synthesis_params(
    profile: ThresholdProfile,
    pattern: str,
    clip_frame_count: int,
    frame_height: int,
    frame_width: int,
    fps: float,
    rng: np.random.Generator,
) -> SynthParams:
    if pattern not in ("general", "red"):
        raise ValueError(f"unknown pattern: {pattern!r}")

    window_frames = max(1, round(fps))
    period_frames = _period_frames(profile, fps)
    # window_frames of ramp-up (counter filling) + 4 periods of guaranteed
    # steady-state tail once the trailing window is fully inside the pattern.
    duration_frames = window_frames + 4 * period_frames

    if clip_frame_count < duration_frames + 1:
        raise ValueError(
            f"clip has {clip_frame_count} frames, need at least "
            f"{duration_frames + 1} for fps={fps} and profile {profile.name!r}"
        )

    latest_start = clip_frame_count - duration_frames
    start_frame = int(rng.integers(1, latest_start + 1))
    end_frame = start_frame + duration_frames - 1

    mask_top, mask_left, mask_height, mask_width = _mask_dims(
        frame_height, frame_width, profile.max_area_ratio + _AREA_MARGIN
    )

    window = InjectionWindow(
        start_frame=start_frame,
        end_frame=end_frame,
        mask_top=mask_top,
        mask_left=mask_left,
        mask_height=mask_height,
        mask_width=mask_width,
        period_frames=period_frames,
        ramp_frames=window_frames,
    )

    if pattern == "general":
        dark_target = profile.general_flash_dark_threshold * 0.5
        bright_target = min(1.0, dark_target + profile.general_flash_delta_threshold + 0.1)
        return SynthParams(pattern="general", window=window, dark_target=dark_target, bright_target=bright_target)

    target_ratio = min(_MAX_SATURATION_TARGET, profile.red_saturation_ratio_threshold + _RATIO_MARGIN)
    r = 0.9
    gb_sum = r / target_ratio - r
    red_rgb = (r, gb_sum / 2, gb_sum / 2)
    baseline_rgb = (0.3, 0.3, 0.3)
    return SynthParams(pattern="red", window=window, red_rgb=red_rgb, baseline_rgb=baseline_rgb)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest flicker-guard/training/tests/test_params.py -v`
Expected: PASS (12 passed — 6 own tests + 6 parametrized shipped-profile cases)

- [ ] **Step 5: Commit**

```bash
git add flicker-guard/training/params.py flicker-guard/training/tests/test_params.py
git commit -m "feat(dataset-synth): add per-profile synthesis parameter sampling"
```

---

## Task 2: Pixel-space flicker injection

**Files:**
- Create: `flicker-guard/training/injection.py`
- Test: `flicker-guard/training/tests/test_injection.py`

**Interfaces:**
- Consumes: `InjectionWindow` from Task 1 (as a plain argument; this module does not import `training.params`... it does need the type for annotations, see below).
- Produces: `inject_general_flash(frames: list[np.ndarray], window: InjectionWindow, dark_target: float, bright_target: float) -> list[np.ndarray]`, `inject_red_flash(frames: list[np.ndarray], window: InjectionWindow, red_rgb: tuple[float, float, float], baseline_rgb: tuple[float, float, float]) -> list[np.ndarray]`. Both return a **new** list; the input `frames` list and its arrays are never mutated.

- [ ] **Step 1: Write the failing test**

```python
# flicker-guard/training/tests/test_injection.py
import numpy as np

from detector.flash import red_flash_mask, transition_mask
from detector.luminance import relative_luminance
from training.injection import inject_general_flash, inject_red_flash
from training.params import InjectionWindow


def _solid_frames(n, h, w, value):
    return [np.full((h, w, 3), value, dtype=np.float32) for _ in range(n)]


def test_inject_general_flash_leaves_frames_outside_window_untouched():
    frames = _solid_frames(10, 20, 20, 0.5)
    window = InjectionWindow(start_frame=3, end_frame=6, mask_top=0, mask_left=0, mask_height=20, mask_width=20, period_frames=1, ramp_frames=1)
    out = inject_general_flash(frames, window, dark_target=0.1, bright_target=0.5)
    assert np.array_equal(out[0], frames[0])
    assert np.array_equal(out[2], frames[2])
    assert np.array_equal(out[7], frames[7])
    assert np.array_equal(out[9], frames[9])


def test_inject_general_flash_leaves_pixels_outside_mask_untouched():
    frames = _solid_frames(6, 20, 20, 0.5)
    window = InjectionWindow(start_frame=1, end_frame=4, mask_top=5, mask_left=5, mask_height=5, mask_width=5, period_frames=1, ramp_frames=1)
    out = inject_general_flash(frames, window, dark_target=0.1, bright_target=0.5)
    assert np.array_equal(out[2][0:5, :, :], frames[2][0:5, :, :])
    assert frames[2][0, 0, 0] == 0.5  # input untouched


def test_inject_general_flash_transitions_are_flagged_by_detector():
    frames = _solid_frames(8, 20, 20, 0.5)
    window = InjectionWindow(start_frame=1, end_frame=6, mask_top=0, mask_left=0, mask_height=20, mask_width=20, period_frames=1, ramp_frames=1)
    out = inject_general_flash(frames, window, dark_target=0.4, bright_target=0.6)
    lum2 = relative_luminance(out[2])
    lum3 = relative_luminance(out[3])
    mask = transition_mask(lum2, lum3, dark_threshold=0.80, delta_threshold=0.10)
    assert mask.all()


def test_inject_red_flash_transitions_are_flagged_by_detector():
    frames = _solid_frames(6, 10, 10, 0.3)
    window = InjectionWindow(start_frame=1, end_frame=4, mask_top=0, mask_left=0, mask_height=10, mask_width=10, period_frames=1, ramp_frames=1)
    out = inject_red_flash(frames, window, red_rgb=(0.9, 0.05, 0.05), baseline_rgb=(0.3, 0.3, 0.3))
    mask = red_flash_mask(out[2], out[3], saturation_ratio_threshold=0.80)
    assert mask.all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest flicker-guard/training/tests/test_injection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.injection'`

- [ ] **Step 3: Write minimal implementation**

```python
# flicker-guard/training/injection.py
"""Pixel-space PSE flicker injection. Alternates a rectangular region of a
clip between two fixed states every `window.period_frames` frames,
guaranteeing a transition Detector's own transition_mask/red_flash_mask will
flag (see training.params for how targets are chosen relative to a
profile).

Frames outside the injection window, and pixels outside the mask, are left
byte-identical to the input -- this is what makes the input clip usable
directly as Mitigator's clean ground truth outside the injected region.
"""
import numpy as np

from training.params import InjectionWindow

# WCAG/BT.709 sRGB encode, the algebraic inverse of detector.luminance's
# decode for the gray case (R=G=B=c): since the BT.709 coefficients sum to
# 1.0, relative_luminance(c, c, c) == decode(c) exactly.
_SRGB_A = 0.055
_LINEAR_CUTOFF = 0.0031308  # 0.04045 / 12.92


def _gray_channel_for_luminance(target_luminance: float) -> float:
    target_luminance = float(np.clip(target_luminance, 0.0, 1.0))
    if target_luminance <= _LINEAR_CUTOFF:
        return target_luminance * 12.92
    return (1 + _SRGB_A) * (target_luminance ** (1 / 2.4)) - _SRGB_A


def _is_pulse_frame(offset: int, period_frames: int) -> bool:
    return (offset // period_frames) % 2 == 1


def _mask_slice(window: InjectionWindow) -> tuple[slice, slice]:
    return (
        slice(window.mask_top, window.mask_top + window.mask_height),
        slice(window.mask_left, window.mask_left + window.mask_width),
    )


def inject_general_flash(
    frames: list[np.ndarray],
    window: InjectionWindow,
    dark_target: float,
    bright_target: float,
) -> list[np.ndarray]:
    dark_channel = _gray_channel_for_luminance(dark_target)
    bright_channel = _gray_channel_for_luminance(bright_target)
    rows, cols = _mask_slice(window)

    out = [frame.copy() for frame in frames]
    for i in range(window.start_frame, window.end_frame + 1):
        offset = i - window.start_frame
        channel = bright_channel if _is_pulse_frame(offset, window.period_frames) else dark_channel
        out[i][rows, cols, :] = channel
    return out


def inject_red_flash(
    frames: list[np.ndarray],
    window: InjectionWindow,
    red_rgb: tuple[float, float, float],
    baseline_rgb: tuple[float, float, float],
) -> list[np.ndarray]:
    rows, cols = _mask_slice(window)

    out = [frame.copy() for frame in frames]
    for i in range(window.start_frame, window.end_frame + 1):
        offset = i - window.start_frame
        color = red_rgb if _is_pulse_frame(offset, window.period_frames) else baseline_rgb
        out[i][rows, cols, 0] = color[0]
        out[i][rows, cols, 1] = color[1]
        out[i][rows, cols, 2] = color[2]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest flicker-guard/training/tests/test_injection.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add flicker-guard/training/injection.py flicker-guard/training/tests/test_injection.py
git commit -m "feat(dataset-synth): add pixel-space flash injection"
```

---

## Task 3: Synthesize-and-validate with retry

**Files:**
- Create: `flicker-guard/training/synth.py`
- Test: `flicker-guard/training/tests/test_synth.py`

**Interfaces:**
- Consumes: `InjectionWindow`, `SynthParams`, `sample_synthesis_params` (Task 1); `inject_general_flash`, `inject_red_flash` (Task 2); `detector.pipeline.run_detection`; `detector.profiles.ThresholdProfile`; `detector.segments.RiskSegment`.
- Produces: `SynthesizedSample` dataclass (`pattern: str`, `profile_name: str`, `window: InjectionWindow`, `clean_frames: list[np.ndarray]`, `degraded_frames: list[np.ndarray]`, `segments: list[RiskSegment]`), `synthesize_sample(clean_frames: list[np.ndarray], fps: float, profile: ThresholdProfile, pattern: str, rng: np.random.Generator, max_retries: int = 5) -> SynthesizedSample | None`.

Validation runs `run_detection` with `margin_seconds=0.0` — Detector's default 0.5s margin pads segments on both ends, which would make almost any segment "cover" the window regardless of whether the injected pattern was really what triggered it. Zero margin gives the tight, unpadded risky span, which is the meaningful signal here.

Because Detector's windowed counter needs a full trailing second to leave the "ramp-up" state (see Task 1's `ramp_frames`), the very start of the injected window won't be flagged risky yet — only the *steady-state tail*, `[window.start_frame + window.ramp_frames, window.end_frame]`, is expected to be reliably risky throughout. Validation checks containment of that tail, not the whole window.

- [ ] **Step 1: Write the failing test**

```python
# flicker-guard/training/tests/test_synth.py
import numpy as np

from detector.profiles import ThresholdProfile
from training.synth import synthesize_sample

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest flicker-guard/training/tests/test_synth.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.synth'`

- [ ] **Step 3: Write minimal implementation**

```python
# flicker-guard/training/synth.py
"""Orchestrates one synthesis attempt: sample parameters, inject the
pattern, and re-validate with the real Detector pipeline before accepting.
Rejected attempts are retried with fresh parameters up to `max_retries`
times -- this is the "confirmation, not blind generation" loop the spec
calls for."""
from dataclasses import dataclass

import numpy as np

from detector.pipeline import run_detection
from detector.profiles import ThresholdProfile
from detector.segments import RiskSegment
from training.injection import inject_general_flash, inject_red_flash
from training.params import InjectionWindow, SynthParams, sample_synthesis_params


@dataclass
class SynthesizedSample:
    pattern: str
    profile_name: str
    window: InjectionWindow
    clean_frames: list[np.ndarray]
    degraded_frames: list[np.ndarray]
    segments: list[RiskSegment]


def _window_is_covered(window: InjectionWindow, segments: list[RiskSegment]) -> bool:
    core_start = window.start_frame + window.ramp_frames
    return any(
        segment.start_frame <= core_start and segment.end_frame >= window.end_frame
        for segment in segments
    )


def _apply_pattern(clean_frames: list[np.ndarray], params: SynthParams) -> list[np.ndarray]:
    if params.pattern == "general":
        return inject_general_flash(clean_frames, params.window, params.dark_target, params.bright_target)
    return inject_red_flash(clean_frames, params.window, params.red_rgb, params.baseline_rgb)


def synthesize_sample(
    clean_frames: list[np.ndarray],
    fps: float,
    profile: ThresholdProfile,
    pattern: str,
    rng: np.random.Generator,
    max_retries: int = 5,
) -> SynthesizedSample | None:
    frame_height, frame_width = clean_frames[0].shape[:2]

    for _ in range(max_retries):
        params = sample_synthesis_params(
            profile, pattern, len(clean_frames), frame_height, frame_width, fps, rng
        )
        degraded_frames = _apply_pattern(clean_frames, params)
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

Run: `pytest flicker-guard/training/tests/test_synth.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add flicker-guard/training/synth.py flicker-guard/training/tests/test_synth.py
git commit -m "feat(dataset-synth): add synthesize-and-validate retry loop"
```

---

## Task 4: Sample writer with resume support

**Files:**
- Create: `flicker-guard/training/dataset_writer.py`
- Test: `flicker-guard/training/tests/test_dataset_writer.py`

**Interfaces:**
- Consumes: `SynthesizedSample` (Task 3).
- Produces: `sample_id(clip_id: str, profile_name: str, pattern: str, index: int) -> str`, `sample_exists(out_root: Path, sid: str) -> bool`, `write_sample(sample: SynthesizedSample, out_root: Path, clip_id: str, index: int) -> str` (returns the sample id).

- [ ] **Step 1: Write the failing test**

```python
# flicker-guard/training/tests/test_dataset_writer.py
import json

import numpy as np

from detector.segments import RiskSegment
from training.dataset_writer import sample_exists, sample_id, write_sample
from training.params import InjectionWindow
from training.synth import SynthesizedSample


def _sample():
    window = InjectionWindow(
        start_frame=1, end_frame=10, mask_top=0, mask_left=0,
        mask_height=4, mask_width=4, period_frames=1, ramp_frames=2,
    )
    frames = [np.full((4, 4, 3), 0.5, dtype=np.float32) for _ in range(12)]
    return SynthesizedSample(
        pattern="general", profile_name="kr", window=window,
        clean_frames=frames, degraded_frames=frames,
        segments=[RiskSegment(start_frame=3, end_frame=10)],
    )


def test_sample_id_format():
    assert sample_id("bear", "kr", "general", 0) == "bear__kr__general__000"


def test_write_sample_creates_frame_files_and_metadata(tmp_path):
    sid = write_sample(_sample(), tmp_path, clip_id="bear", index=0)
    sample_dir = tmp_path / sid
    assert (sample_dir / "clean" / "000000.png").exists()
    assert (sample_dir / "clean" / "000011.png").exists()
    assert (sample_dir / "degraded" / "000000.png").exists()
    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["clip_id"] == "bear"
    assert meta["profile"] == "kr"
    assert meta["pattern"] == "general"
    assert meta["injected_window"]["start_frame"] == 1
    assert meta["segments"] == [{"start_frame": 3, "end_frame": 10}]


def test_sample_exists_true_after_write_false_before(tmp_path):
    sid = sample_id("bear", "kr", "general", 0)
    assert not sample_exists(tmp_path, sid)
    write_sample(_sample(), tmp_path, clip_id="bear", index=0)
    assert sample_exists(tmp_path, sid)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest flicker-guard/training/tests/test_dataset_writer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.dataset_writer'`

- [ ] **Step 3: Write minimal implementation**

```python
# flicker-guard/training/dataset_writer.py
"""Writes an accepted SynthesizedSample to disk as a clean/degraded PNG
frame-sequence pair plus a metadata JSON, and supports resuming an
interrupted batch by skipping samples that already exist."""
import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np

from training.synth import SynthesizedSample


def sample_id(clip_id: str, profile_name: str, pattern: str, index: int) -> str:
    return f"{clip_id}__{profile_name}__{pattern}__{index:03d}"


def sample_exists(out_root: Path, sid: str) -> bool:
    return (Path(out_root) / sid / "meta.json").exists()


def _write_frames(frames: list[np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame_rgb in enumerate(frames):
        frame_bgr_uint8 = cv2.cvtColor(
            np.clip(frame_rgb * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        cv2.imwrite(str(out_dir / f"{i:06d}.png"), frame_bgr_uint8)


def write_sample(
    sample: SynthesizedSample,
    out_root: Path,
    clip_id: str,
    index: int,
) -> str:
    sid = sample_id(clip_id, sample.profile_name, sample.pattern, index)
    sample_dir = Path(out_root) / sid
    _write_frames(sample.clean_frames, sample_dir / "clean")
    _write_frames(sample.degraded_frames, sample_dir / "degraded")

    meta = {
        "sample_id": sid,
        "clip_id": clip_id,
        "profile": sample.profile_name,
        "pattern": sample.pattern,
        "injected_window": dataclasses.asdict(sample.window),
        "segments": [dataclasses.asdict(s) for s in sample.segments],
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return sid
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest flicker-guard/training/tests/test_dataset_writer.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add flicker-guard/training/dataset_writer.py flicker-guard/training/tests/test_dataset_writer.py
git commit -m "feat(dataset-synth): add sample writer with resume support"
```

---

## Task 5: Batch CLI

**Files:**
- Create: `flicker-guard/training/cli.py`
- Test: `flicker-guard/training/tests/test_cli.py`

**Interfaces:**
- Consumes: `detector.cli.read_video_frames`, `detector.cli.VideoReadError`, `detector.profiles.load_profile`, `synthesize_sample` (Task 3), `sample_id`, `sample_exists`, `write_sample` (Task 4).
- Produces: `run_batch(clips_dir: Path, profiles_dir: Path, out_root: Path, samples_per_combo: int, seed: int = 0) -> dict` (summary dict), `main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# flicker-guard/training/tests/test_cli.py
import json

import cv2
import numpy as np

from training.cli import main, run_batch

PROFILE_JSON = {
    "name": "tiny",
    "max_flashes_per_second": 3,
    "max_area_ratio": 0.10,
    "general_flash_dark_threshold": 0.80,
    "general_flash_delta_threshold": 0.10,
    "red_saturation_ratio_threshold": 0.80,
}


def _write_clip(path, n_frames=40, fps=10, size=(32, 32)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    frame = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()


def _write_profiles_dir(dir_path):
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "tiny.json").write_text(json.dumps(PROFILE_JSON), encoding="utf-8")


def test_run_batch_creates_accepted_samples(tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _write_clip(clips_dir / "bear.mp4")
    profiles_dir = tmp_path / "profiles"
    _write_profiles_dir(profiles_dir)
    out_dir = tmp_path / "out"

    summary = run_batch(clips_dir, profiles_dir, out_dir, samples_per_combo=1, seed=0)

    assert summary["accepted"] == 2  # general + red, one profile
    assert (out_dir / "bear__tiny__general__000" / "meta.json").exists()
    assert (out_dir / "bear__tiny__red__000" / "meta.json").exists()


def test_run_batch_skips_existing_samples_on_rerun(tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _write_clip(clips_dir / "bear.mp4")
    profiles_dir = tmp_path / "profiles"
    _write_profiles_dir(profiles_dir)
    out_dir = tmp_path / "out"

    run_batch(clips_dir, profiles_dir, out_dir, samples_per_combo=1, seed=0)
    second = run_batch(clips_dir, profiles_dir, out_dir, samples_per_combo=1, seed=1)

    assert second["accepted"] == 0
    assert second["skipped_existing"] == 2


def test_run_batch_records_unreadable_clip(tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "broken.mp4").write_bytes(b"not a real video")
    profiles_dir = tmp_path / "profiles"
    _write_profiles_dir(profiles_dir)
    out_dir = tmp_path / "out"

    summary = run_batch(clips_dir, profiles_dir, out_dir, samples_per_combo=1, seed=0)

    assert summary["accepted"] == 0
    assert summary["failed"] == 1
    assert summary["failed_details"][0]["reason"] == "unreadable_clip"


def test_main_writes_summary_json(tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _write_clip(clips_dir / "bear.mp4")
    profiles_dir = tmp_path / "profiles"
    _write_profiles_dir(profiles_dir)
    out_dir = tmp_path / "out"

    exit_code = main([
        "--clips-dir", str(clips_dir),
        "--profiles-dir", str(profiles_dir),
        "--output", str(out_dir),
    ])

    assert exit_code == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["accepted"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest flicker-guard/training/tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.cli'`

- [ ] **Step 3: Write minimal implementation**

```python
# flicker-guard/training/cli.py
"""Batch driver: synthesizes PSE training samples from a directory of clean
clips against every profile JSON in --profiles-dir and both flash patterns,
writing accepted samples under --output and skipping ones that already
exist there (safe to re-run after an interrupted batch)."""
import argparse
import json
from pathlib import Path

import numpy as np

from detector.cli import VideoReadError, read_video_frames
from detector.profiles import load_profile
from training.dataset_writer import sample_exists, sample_id, write_sample
from training.synth import synthesize_sample

PATTERNS = ("general", "red")


def _iter_clip_paths(clips_dir: Path):
    return sorted(Path(clips_dir).glob("*.mp4"))


def run_batch(
    clips_dir: Path,
    profiles_dir: Path,
    out_root: Path,
    samples_per_combo: int,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    profile_paths = sorted(Path(profiles_dir).glob("*.json"))
    profiles = [load_profile(p) for p in profile_paths]

    summary = {"accepted": 0, "skipped_existing": 0, "failed": 0, "failed_details": []}

    for clip_path in _iter_clip_paths(clips_dir):
        clip_id = clip_path.stem
        try:
            frame_iter, fps = read_video_frames(str(clip_path))
            clean_frames = list(frame_iter)
        except VideoReadError as exc:
            summary["failed"] += 1
            summary["failed_details"].append(
                {"clip_id": clip_id, "reason": "unreadable_clip", "error": str(exc)}
            )
            continue

        for profile in profiles:
            for pattern in PATTERNS:
                for index in range(samples_per_combo):
                    sid = sample_id(clip_id, profile.name, pattern, index)
                    if sample_exists(out_root, sid):
                        summary["skipped_existing"] += 1
                        continue
                    try:
                        sample = synthesize_sample(clean_frames, fps, profile, pattern, rng)
                    except ValueError as exc:
                        summary["failed"] += 1
                        summary["failed_details"].append(
                            {"clip_id": clip_id, "profile": profile.name, "pattern": pattern,
                             "index": index, "reason": "clip_too_short", "error": str(exc)}
                        )
                        continue
                    if sample is None:
                        summary["failed"] += 1
                        summary["failed_details"].append(
                            {"clip_id": clip_id, "profile": profile.name, "pattern": pattern,
                             "index": index, "reason": "validation_exhausted"}
                        )
                        continue
                    write_sample(sample, out_root, clip_id, index)
                    summary["accepted"] += 1

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthesize PSE training samples from clean clips.")
    parser.add_argument("--clips-dir", required=True)
    parser.add_argument("--profiles-dir", default="configs/profiles")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-combo", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary-output")
    args = parser.parse_args(argv)

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = run_batch(
        clips_dir=Path(args.clips_dir),
        profiles_dir=Path(args.profiles_dir),
        out_root=out_root,
        samples_per_combo=args.samples_per_combo,
        seed=args.seed,
    )

    summary_path = Path(args.summary_output) if args.summary_output else out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest flicker-guard/training/tests/test_cli.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add flicker-guard/training/cli.py flicker-guard/training/tests/test_cli.py
git commit -m "feat(dataset-synth): add batch CLI over clips x profiles x patterns"
```

---

## Task 6: Package setup and full-suite verification

**Files:**
- Create: `flicker-guard/training/__init__.py` (empty)
- Create: `flicker-guard/training/tests/__init__.py` (empty)

- [ ] **Step 1: Add package markers**

```python
# flicker-guard/training/__init__.py
```

```python
# flicker-guard/training/tests/__init__.py
```

- [ ] **Step 2: Run the full repo test suite**

Run: `pytest flicker-guard -v` (from the repo root; `pytest.ini`'s `pythonpath = .` already covers `training/` the same way it covers `detector/`, no config change needed)
Expected: All `detector/tests/` (66) and `training/tests/` (Task 1: 12, Task 2: 4, Task 3: 4, Task 4: 3, Task 5: 4 = 27) tests PASS — 93 passed total, no new dependency required (`requirements.txt` already lists numpy/opencv-python/pytest).

- [ ] **Step 3: Commit**

```bash
git add flicker-guard/training/__init__.py flicker-guard/training/tests/__init__.py
git commit -m "chore(dataset-synth): add training package markers"
```
