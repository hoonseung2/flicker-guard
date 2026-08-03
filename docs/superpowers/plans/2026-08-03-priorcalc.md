# PriorCalc Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build PriorCalc — computes a temporally-smoothed "target" illumination histogram plus a per-pixel risk mask for every frame, reusing Detector's own risk determination instead of reimplementing BlazeBVD's exposure-map/frame-classification logic, so Mitigator has conditioning material for "what should this frame's brightness look like" and "where exactly is the hazard."

**Architecture:** `detector/pipeline.py` gets a small, additive extension (`run_detection_with_masks`) that exposes the per-pixel flash mask it already computes internally and currently discards — `run_detection`'s existing signature and behavior are untouched. `prior/histogram.py` computes per-frame illumination histograms and a stateful trailing-window smoother that excludes Detector-flagged frames from the average. `prior/compute.py` wires the two together.

**Tech Stack:** Python 3.10+, numpy. No new dependencies.

## Global Constraints

- "Which frames are risky" is never recomputed independently — always sourced from Detector's `RiskSegment`/`FlickerScore.uncertain` (per design spec's decision to reuse Detector over reimplementing BlazeBVD's singular-frames-set).
- "Which pixels are risky" is never recomputed independently — always sourced from Detector's existing `transition_mask`/`red_flash_mask` output, exposed via `run_detection_with_masks` (no new BlazeBVD-style exposure-map algorithm).
- `detector.pipeline.run_detection`'s existing signature, return type, and behavior must not change in any way — all 66 existing Detector tests and all 27 existing `training/` tests must keep passing unmodified.
- A frame counts as excluded from temporal smoothing if `frame_index` falls inside any returned `RiskSegment`, **or** its `FlickerScore.uncertain` is `True` — either condition alone is sufficient.
- No new dependencies beyond numpy (already used throughout `detector/`).
- The automated test suite uses only small in-memory synthetic frames — no real DAVIS data required to run `pytest`.

---

## File Structure

```
flicker-guard/
├── detector/
│   └── pipeline.py         # Task 1 (modified — additive only)
├── prior/
│   ├── __init__.py         # Task 4
│   ├── histogram.py        # Task 2
│   ├── compute.py          # Task 3
│   └── tests/
│       ├── __init__.py     # Task 4
│       ├── test_histogram.py
│       └── test_compute.py
```

---

## Task 1: Expose Detector's per-pixel mask via `run_detection_with_masks`

**Files:**
- Modify: `flicker-guard/detector/pipeline.py`
- Test: `flicker-guard/detector/tests/test_pipeline.py` (add tests, keep all 7 existing ones unmodified)

**Interfaces:**
- Consumes: nothing new — same imports `detector.pipeline` already has (`detector.flash`, `detector.luminance`, `detector.motion`, `detector.profiles.ThresholdProfile`, `detector.scoring.{FlickerScore, WindowedFlashCounter}`, `detector.segments.{RiskSegment, scores_to_segments}`).
- Produces: `run_detection_with_masks(frames: Iterable[np.ndarray], fps: float, profile: ThresholdProfile, margin_seconds: float = 0.5) -> tuple[list[FlickerScore], list[RiskSegment], list[np.ndarray]]` — identical to `run_detection` except it also returns the list of per-frame `(H, W)` bool masks. `run_detection` itself is refactored to share the same internal per-frame loop, so both functions are behaviorally guaranteed identical for the `scores`/`segments` they produce.

- [ ] **Step 1: Write the failing tests**

Add these to `flicker-guard/detector/tests/test_pipeline.py`, alongside the existing 7 tests and the existing `PROFILE`/`_solid_frame` fixtures already in that file (do not remove or modify anything already there):

```python
# add to flicker-guard/detector/tests/test_pipeline.py
from detector.flash import red_flash_mask, transition_mask
from detector.luminance import relative_luminance
from detector.motion import compensate_shift, estimate_global_shift
from detector.pipeline import run_detection_with_masks  # add to existing run_detection import


def test_run_detection_with_masks_matches_run_detection_scores_and_segments():
    dark, bright = _solid_frame(0.05), _solid_frame(0.95)
    frames = [dark, bright] * 8
    scores_a, segments_a = run_detection(frames, fps=10, profile=PROFILE, margin_seconds=0.0)
    scores_b, segments_b, masks = run_detection_with_masks(frames, fps=10, profile=PROFILE, margin_seconds=0.0)
    assert scores_a == scores_b
    assert segments_a == segments_b
    assert len(masks) == len(frames)
    assert masks[0].shape == (20, 20)


def test_run_detection_with_masks_frame_zero_is_all_true():
    frames = [_solid_frame(0.5)] * 5
    _, _, masks = run_detection_with_masks(frames, fps=10, profile=PROFILE, margin_seconds=0.0)
    assert masks[0].all()


def test_run_detection_with_masks_matches_manual_mask_computation():
    dark, bright = _solid_frame(0.05), _solid_frame(0.95)
    frames = [dark, bright, dark]
    _, _, masks = run_detection_with_masks(frames, fps=10, profile=PROFILE, margin_seconds=0.0)

    lum0 = relative_luminance(frames[0])
    lum1 = relative_luminance(frames[1])
    dx, dy = estimate_global_shift(lum0, lum1)
    aligned_prev_lum = compensate_shift(lum0, dx, dy)
    aligned_prev_rgb = compensate_shift(frames[0], dx, dy)
    expected_mask = transition_mask(
        aligned_prev_lum, lum1,
        dark_threshold=PROFILE.general_flash_dark_threshold,
        delta_threshold=PROFILE.general_flash_delta_threshold,
    ) | red_flash_mask(
        aligned_prev_rgb, frames[1],
        saturation_ratio_threshold=PROFILE.red_saturation_ratio_threshold,
    )
    assert (masks[1] == expected_mask).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest detector/tests/test_pipeline.py -v`
Expected: the 3 new tests FAIL with `ImportError: cannot import name 'run_detection_with_masks'`; the 7 pre-existing tests still PASS (confirms you haven't broken anything before you've even started).

- [ ] **Step 3: Refactor `run_detection` and add `run_detection_with_masks`**

Replace the entire contents of `flicker-guard/detector/pipeline.py` with:

```python
# flicker-guard/detector/pipeline.py
"""Wires the luminance / motion / flash / scoring / segment modules into a
single call that scans a full frame sequence.

**MVP limitations — external validation required.** This detector is a
simplified approximation of the WCAG / Harding General Flash and Red Flash
Threshold Algorithm, not an implementation of it:

- Flash counting is flagged-frame counting over a trailing one-second window,
  not full opposing-transition-pair analysis; the reported count runs at
  roughly 2x the true visual flash rate (conservative, see
  `detector/scoring.py`).
- `ThresholdProfile` encodes only a flash-frequency limit and a flagged-area
  limit. Ofcom's dark-scene sub-rule (luminance < 160 with contrast >= 20),
  Japan's high-contrast pattern-density rule, and WCAG's "25% of any 10-degree
  visual field" area sub-clause (see README section 9) are **not** encoded
  and are therefore **not** detected.
- Motion compensation is global/pan-only; local object motion is not
  compensated and leaves a small residual flagged area at frame borders.

Per README section 9, Harding FPA or equivalent external validation is still
required before production use. Nothing here may be presented as a guarantee
that content is safe.

Streaming (final-review finding I6): `frames` is any iterable — the pipeline
is strictly causal and never holds more than the previous and current frame,
so a generator from `detector.cli.read_video_frames` streams a video without
materialising it in RAM.

Uncertain edges (final-review finding I4, README section 7): frame 0 has no
predecessor, so its transition is unknown rather than safe. It is fed to the
counter as a fully flagged, explicitly unmeasured frame, which can only push
the verdict toward "risky".

Per-pixel masks (PriorCalc plan): `run_detection` computes a per-pixel flash
mask for every frame internally and discards it, keeping only the aggregate
`FlickerScore`. `run_detection_with_masks` is the same detection, sharing the
same internal per-frame loop (`_iter_scores_and_masks`), but also returns
those masks -- PriorCalc needs to know *where* in the frame a transition
happened, not just whether the frame is risky. `run_detection`'s own
signature and behavior are unchanged by this addition.
"""
from collections.abc import Iterable, Iterator

import numpy as np

from detector.flash import red_flash_mask, transition_mask
from detector.luminance import relative_luminance
from detector.motion import compensate_shift, estimate_global_shift
from detector.profiles import ThresholdProfile
from detector.scoring import FlickerScore, WindowedFlashCounter
from detector.segments import RiskSegment, scores_to_segments


def _iter_scores_and_masks(
    frames: Iterable[np.ndarray],
    fps: float,
    profile: ThresholdProfile,
) -> Iterator[tuple[FlickerScore, np.ndarray]]:
    """Shared per-frame computation behind run_detection and
    run_detection_with_masks -- the single source of truth both public
    functions delegate to, so they can never behaviorally diverge. Not
    part of the public API."""
    counter = WindowedFlashCounter(fps=fps)

    prev_rgb = None
    prev_luminance = None
    for i, frame in enumerate(frames):
        curr_luminance = relative_luminance(frame)
        if prev_rgb is None:
            # No predecessor: the transition into this frame is unknown, so
            # assume the worst instead of fabricating an all-safe mask (I4).
            mask = np.ones(curr_luminance.shape, dtype=bool)
            uncertain = True
        else:
            dx, dy = estimate_global_shift(prev_luminance, curr_luminance)
            aligned_prev_luminance = compensate_shift(prev_luminance, dx, dy)
            aligned_prev_rgb = compensate_shift(prev_rgb, dx, dy)
            mask = transition_mask(
                aligned_prev_luminance,
                curr_luminance,
                dark_threshold=profile.general_flash_dark_threshold,
                delta_threshold=profile.general_flash_delta_threshold,
            ) | red_flash_mask(
                aligned_prev_rgb,
                frame,
                saturation_ratio_threshold=profile.red_saturation_ratio_threshold,
            )
            uncertain = False
        score = counter.update(i, mask, uncertain=uncertain)
        yield score, mask
        prev_rgb, prev_luminance = frame, curr_luminance


def run_detection(
    frames: Iterable[np.ndarray],
    fps: float,
    profile: ThresholdProfile,
    margin_seconds: float = 0.5,
) -> tuple[list[FlickerScore], list[RiskSegment]]:
    scores = [score for score, _mask in _iter_scores_and_masks(frames, fps, profile)]
    margin_frames = round(margin_seconds * fps)
    segments = scores_to_segments(scores, profile, margin_frames, total_frames=len(scores))
    return scores, segments


def run_detection_with_masks(
    frames: Iterable[np.ndarray],
    fps: float,
    profile: ThresholdProfile,
    margin_seconds: float = 0.5,
) -> tuple[list[FlickerScore], list[RiskSegment], list[np.ndarray]]:
    scores: list[FlickerScore] = []
    masks: list[np.ndarray] = []
    for score, mask in _iter_scores_and_masks(frames, fps, profile):
        scores.append(score)
        masks.append(mask)
    margin_frames = round(margin_seconds * fps)
    segments = scores_to_segments(scores, profile, margin_frames, total_frames=len(scores))
    return scores, segments, masks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest detector/tests/test_pipeline.py -v`
Expected: PASS (10 passed — 7 pre-existing + 3 new)

Then run the full Detector suite to confirm nothing else broke:
Run: `pytest detector/tests/ -v`
Expected: PASS (69 passed — 66 pre-existing + 3 new)

- [ ] **Step 5: Commit**

```bash
git add flicker-guard/detector/pipeline.py flicker-guard/detector/tests/test_pipeline.py
git commit -m "feat(detector): expose per-pixel flash mask via run_detection_with_masks"
```

---

## Task 2: Illumination histogram + temporal smoothing

**Files:**
- Create: `flicker-guard/prior/histogram.py`
- Test: `flicker-guard/prior/tests/test_histogram.py`

**Interfaces:**
- Consumes: nothing from other new modules (pure numpy).
- Produces: `compute_illumination_histogram(luminance: np.ndarray, n_bins: int = 64) -> np.ndarray` (shape `(n_bins,)`, sums to 1.0), `TargetHistogramSmoother(window_frames: int)` class with `update(histogram: np.ndarray, is_risky_or_uncertain: bool) -> np.ndarray | None`.

- [ ] **Step 1: Write the failing test**

```python
# flicker-guard/prior/tests/test_histogram.py
import numpy as np

from prior.histogram import TargetHistogramSmoother, compute_illumination_histogram


def test_compute_illumination_histogram_sums_to_one():
    rng = np.random.default_rng(0)
    luminance = rng.random((10, 10)).astype(np.float32)
    hist = compute_illumination_histogram(luminance, n_bins=64)
    assert hist.shape == (64,)
    assert np.isclose(hist.sum(), 1.0)


def test_compute_illumination_histogram_all_zero_frame_fills_first_bin():
    luminance = np.zeros((5, 5), dtype=np.float32)
    hist = compute_illumination_histogram(luminance, n_bins=64)
    assert hist[0] == 1.0
    assert hist[1:].sum() == 0.0


def test_compute_illumination_histogram_half_and_half():
    luminance = np.zeros((4, 4), dtype=np.float32)
    luminance[:2, :] = 1.0  # half at 0.0 (bin 0), half at 1.0 (last bin)
    hist = compute_illumination_histogram(luminance, n_bins=4)
    assert np.isclose(hist[0], 0.5)
    assert np.isclose(hist[-1], 0.5)


def test_target_histogram_smoother_averages_only_clean_frames_in_window():
    smoother = TargetHistogramSmoother(window_frames=4)
    clean_hist = np.array([1.0, 0.0, 0.0, 0.0])
    flicker_hist = np.array([0.0, 0.0, 0.0, 1.0])

    smoother.update(clean_hist, is_risky_or_uncertain=False)
    smoother.update(clean_hist, is_risky_or_uncertain=False)
    target = smoother.update(flicker_hist, is_risky_or_uncertain=True)

    assert np.allclose(target, clean_hist)


def test_target_histogram_smoother_falls_back_to_last_clean_when_window_has_none():
    smoother = TargetHistogramSmoother(window_frames=2)
    clean_hist = np.array([1.0, 0.0])
    flicker_hist = np.array([0.0, 1.0])

    smoother.update(clean_hist, is_risky_or_uncertain=False)
    smoother.update(flicker_hist, is_risky_or_uncertain=True)
    target = smoother.update(flicker_hist, is_risky_or_uncertain=True)
    # window (size 2) now holds only the two flicker updates -> no clean frame in window
    assert np.allclose(target, clean_hist)


def test_target_histogram_smoother_returns_none_before_any_clean_frame_seen():
    smoother = TargetHistogramSmoother(window_frames=3)
    flicker_hist = np.array([0.0, 1.0])
    target = smoother.update(flicker_hist, is_risky_or_uncertain=True)
    assert target is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest prior/tests/test_histogram.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prior.histogram'`

- [ ] **Step 3: Write minimal implementation**

```python
# flicker-guard/prior/histogram.py
"""Per-frame illumination histograms and time-axis smoothing.

Reuses Detector's own risk determination (RiskSegment / FlickerScore.uncertain)
instead of an independent flicker-frame classifier: `TargetHistogramSmoother`
just takes a boolean per frame ("is this frame risky or uncertain?") supplied
by the caller (see prior/compute.py) and excludes those frames' histograms
from the trailing-window average -- so the smoothed target represents what
illumination should look like absent the flicker itself, not a blend with it.
"""
import numpy as np


def compute_illumination_histogram(luminance: np.ndarray, n_bins: int = 64) -> np.ndarray:
    histogram, _ = np.histogram(luminance, bins=n_bins, range=(0.0, 1.0))
    return histogram.astype(np.float64) / luminance.size


class TargetHistogramSmoother:
    """Trailing-window average of a histogram sequence, excluding frames the
    caller marks risky/uncertain. Falls back to the most recently seen clean
    frame's histogram when the current window holds no clean frame; returns
    None until at least one clean frame has ever been seen."""

    def __init__(self, window_frames: int):
        self.window_frames = window_frames
        self._history: list[tuple[np.ndarray, bool]] = []
        self._last_clean: np.ndarray | None = None

    def update(self, histogram: np.ndarray, is_risky_or_uncertain: bool) -> np.ndarray | None:
        self._history.append((histogram, is_risky_or_uncertain))
        if len(self._history) > self.window_frames:
            self._history.pop(0)

        if not is_risky_or_uncertain:
            self._last_clean = histogram

        clean_in_window = [h for h, risky in self._history if not risky]
        if clean_in_window:
            return np.mean(clean_in_window, axis=0)
        return self._last_clean
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest prior/tests/test_histogram.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add flicker-guard/prior/histogram.py flicker-guard/prior/tests/test_histogram.py
git commit -m "feat(prior): add illumination histogram and temporal smoothing"
```

---

## Task 3: `compute_prior` orchestrator

**Files:**
- Create: `flicker-guard/prior/compute.py`
- Test: `flicker-guard/prior/tests/test_compute.py`

**Interfaces:**
- Consumes: `run_detection_with_masks` (Task 1); `compute_illumination_histogram`, `TargetHistogramSmoother` (Task 2); `detector.luminance.relative_luminance`, `detector.profiles.ThresholdProfile` (existing Detector).
- Produces: `FramePrior` dataclass (`frame_index: int`, `target_histogram: np.ndarray | None`, `mask: np.ndarray`), `compute_prior(frames: list[np.ndarray], fps: float, profile: ThresholdProfile, n_bins: int = 64) -> list[FramePrior]`.

- [ ] **Step 1: Write the failing test**

```python
# flicker-guard/prior/tests/test_compute.py
import numpy as np

from detector.luminance import relative_luminance
from detector.profiles import ThresholdProfile
from prior.compute import compute_prior
from prior.histogram import compute_illumination_histogram
from training.injection import inject_general_flash
from training.params import InjectionWindow

PROFILE = ThresholdProfile(
    name="test", max_flashes_per_second=3, max_area_ratio=0.10,
    general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
    red_saturation_ratio_threshold=0.80,
)


def _clean_clip(n, h=20, w=20, value=0.5):
    return [np.full((h, w, 3), value, dtype=np.float32) for _ in range(n)]


def test_compute_prior_marks_injected_window_pixels_in_mask():
    clean = _clean_clip(n=30)
    window = InjectionWindow(
        start_frame=5, end_frame=20, mask_top=0, mask_left=0,
        mask_height=20, mask_width=20, period_frames=1, ramp_frames=10,
    )
    degraded = inject_general_flash(clean, window, dark_target=0.2, bright_target=0.8)

    results = compute_prior(degraded, fps=10.0, profile=PROFILE)

    assert len(results) == 30
    assert results[6].mask.all()


def test_compute_prior_target_histogram_returns_to_baseline_well_after_injection():
    clean = _clean_clip(n=80, value=0.5)
    window = InjectionWindow(
        start_frame=10, end_frame=20, mask_top=0, mask_left=0,
        mask_height=20, mask_width=20, period_frames=1, ramp_frames=10,
    )
    degraded = inject_general_flash(clean, window, dark_target=0.2, bright_target=0.8)

    results = compute_prior(degraded, fps=10.0, profile=PROFILE)

    baseline_histogram = compute_illumination_histogram(relative_luminance(clean[0]), n_bins=64)
    late_target = results[75].target_histogram
    assert late_target is not None
    assert np.allclose(late_target, baseline_histogram, atol=1e-6)


def test_compute_prior_returns_frame_prior_per_frame_with_matching_indices():
    clean = _clean_clip(n=10)
    results = compute_prior(clean, fps=10.0, profile=PROFILE)
    assert [r.frame_index for r in results] == list(range(10))
    assert results[0].target_histogram is None  # frame 0 always uncertain, no clean frame seen yet
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest prior/tests/test_compute.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prior.compute'`

- [ ] **Step 3: Write minimal implementation**

```python
# flicker-guard/prior/compute.py
"""Wires Detector's extended run_detection_with_masks together with
prior.histogram: for every frame, produces a (smoothed target illumination
histogram, per-pixel risk mask) pair for Mitigator conditioning.

"Risky" for smoothing purposes uses Detector's *margin-padded* segments
(default margin_seconds=0.5, i.e. run_detection_with_masks's default) rather
than the zero-margin check DatasetSynth's validation uses -- Mitigator will
only ever receive margin-padded segments from BufferManager in production, so
PriorCalc's notion of "risky" should match what Mitigator actually sees.
"""
from dataclasses import dataclass

import numpy as np

from detector.luminance import relative_luminance
from detector.pipeline import run_detection_with_masks
from detector.profiles import ThresholdProfile
from prior.histogram import TargetHistogramSmoother, compute_illumination_histogram


@dataclass
class FramePrior:
    frame_index: int
    target_histogram: np.ndarray | None
    mask: np.ndarray


def compute_prior(
    frames: list[np.ndarray],
    fps: float,
    profile: ThresholdProfile,
    n_bins: int = 64,
) -> list[FramePrior]:
    scores, segments, masks = run_detection_with_masks(frames, fps=fps, profile=profile)

    risky_frame_indices = set()
    for segment in segments:
        risky_frame_indices.update(range(segment.start_frame, segment.end_frame + 1))

    window_frames = max(1, round(fps))
    smoother = TargetHistogramSmoother(window_frames=window_frames)

    results: list[FramePrior] = []
    for frame, score, mask in zip(frames, scores, masks):
        histogram = compute_illumination_histogram(relative_luminance(frame), n_bins=n_bins)
        is_risky_or_uncertain = score.frame_index in risky_frame_indices or score.uncertain
        target = smoother.update(histogram, is_risky_or_uncertain)
        results.append(FramePrior(frame_index=score.frame_index, target_histogram=target, mask=mask))
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest prior/tests/test_compute.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add flicker-guard/prior/compute.py flicker-guard/prior/tests/test_compute.py
git commit -m "feat(prior): add compute_prior orchestrator"
```

---

## Task 4: Package setup and full-suite verification

**Files:**
- Create: `flicker-guard/prior/__init__.py` (empty)
- Create: `flicker-guard/prior/tests/__init__.py` (empty)

- [ ] **Step 1: Add package markers**

```python
# flicker-guard/prior/__init__.py
```

```python
# flicker-guard/prior/tests/__init__.py
```

- [ ] **Step 2: Run the full repo test suite**

Run: `pytest -v` (from the flicker-guard repo root; `pytest.ini`'s `pythonpath = .` already covers `prior/` the same way it covers `detector/` and `training/`, no config change needed)
Expected: All `detector/tests/` (69), `training/tests/` (27), and `prior/tests/` (Task 2: 6, Task 3: 3 = 9) tests PASS — **105 passed total**, no new dependency required.

- [ ] **Step 3: Commit**

```bash
git add flicker-guard/prior/__init__.py flicker-guard/prior/tests/__init__.py
git commit -m "chore(prior): add package markers"
```
