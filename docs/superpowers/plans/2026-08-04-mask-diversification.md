# Injection Placement Fix + Mask Diversification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix DatasetSynth's realistic injection path so (a) the injected risk window is placed to leave PriorCalc enough clean runway to compute a target histogram, and (b) the injected region is one-to-three moving shapes (rectangle/circle/beam) instead of one static centered rectangle.

**Architecture:** `training/params.py`'s `sample_synthesis_params` gets a placement-relaxation fix (shared by flat and realistic modes, no interface change). A new leaf module `training/mask_shapes.py` renders boolean pixel masks for three shape kinds and their analytic pixel areas. `training/params.py` gains a `LightShape` dataclass, a `sample_lights` sampler built on `mask_shapes`, and a new `sample_synthesis_params_realistic` that layers lights onto the existing `InjectionWindow`. `training/injection.py`'s two realistic-mode functions render a per-frame union of the window's lights (linearly interpolating each light's position across the window) instead of a fixed rectangle slice, falling back to the legacy rectangle when a window has no lights (so the flat-mode functions and all existing realistic-mode call sites that construct a plain rectangle window are untouched).

**Tech Stack:** Python 3.10+, numpy (already in requirements.txt). No new dependencies.

## Global Constraints

- **Flat-mode injection (`inject_general_flash`, `inject_red_flash`) is never touched.** Only `inject_general_flash_realistic`/`inject_red_flash_realistic` gain shape/movement support.
- **Backward compatibility is mandatory, not best-effort.** Every existing test in `training/tests/test_injection.py`, `test_params.py`, `test_synth.py`, and `test_dataset_writer.py` that constructs a plain rectangle `InjectionWindow` (no `lights`) must keep passing unmodified, except the one named exception in Task 5 (a test whose old assertion becomes structurally invalid once `synthesize_sample_realistic` starts using lights — see that task).
- **`ClipTooShortError`'s trigger condition does not change.** A clip with `clip_frame_count < duration_frames + 1` still raises it, exactly as before.
- **No new color categories.** Lights use the same `general` (gain-based luminance, `inject_general_flash_realistic`) and `red` (gain-based saturation, `inject_red_flash_realistic`) color logic already shipped — shape/movement changes *where* the gain is applied, never *what* color logic computes.
- **Lights share one `period_frames`/pulse timing** — all lights in a sample turn on/off together (no per-light async phase in this plan).
- **A light's size is fixed for the whole window; only its center position moves** (linear interpolation from a start center to an end center across `[window.start_frame, window.end_frame]`).
- **Area validation is conservative, not exact:** the combined target area for `sample_lights` is computed as if lights never overlap (sum of individual analytic areas), matching how `sample_synthesis_params` already validates the legacy rectangle via `_require_exceeds`.
- Every new file follows the existing project style: no comments explaining *what* code does, only *why* where genuinely non-obvious; module docstring at the top; tests colocated in a `tests/` subpackage.

---

## File Structure

```
flicker-guard/
├── training/
│   ├── params.py               # Task 1 (placement fix), Task 3 (LightShape + sample_lights), Task 5 (sample_synthesis_params_realistic)
│   ├── mask_shapes.py           # Task 2 (new file) — shape rendering + area formulas
│   ├── injection.py             # Task 4 (per-frame light rendering wired into the two realistic functions)
│   ├── synth.py                 # Task 5 (synthesize_sample_realistic uses the new sampler)
│   ├── dataset_writer.py        # unmodified — dataclasses.asdict already serializes the new field
│   └── tests/
│       ├── test_params.py            # Task 1, Task 3, Task 5
│       ├── test_mask_shapes.py       # Task 2 (new file)
│       ├── test_injection.py         # Task 4
│       ├── test_synth.py             # Task 5 (also fixes one pre-existing test — see Task 5)
│       └── test_dataset_writer.py    # Task 5 (lights round-trip through meta.json)
└── docs/superpowers/specs/2026-08-03-mitigator-design.md   # Task 6, manual update after real-data re-measurement
```

---

## Task 1: Relax injection placement to leave PriorCalc a clean runway

**Files:**
- Modify: `training/params.py:110-136` (`sample_synthesis_params`)
- Test: `training/tests/test_params.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `sample_synthesis_params`'s signature and return type (`SynthParams`) are unchanged — only how `start_frame` is chosen inside it changes. `ClipTooShortError`'s trigger condition (`clip_frame_count < duration_frames + 1`) is unchanged.

- [ ] **Step 1: Write the failing tests**

Add to `training/tests/test_params.py` (below the existing tests, same `PROFILE` fixture already defined at the top of the file):

```python
def test_sample_synthesis_params_prefers_a_start_with_full_runway_when_clip_allows_it():
    # fps=24 -> window_frames=24, desired_runway=2*24=48. A 200-frame clip
    # gives latest_start = 200 - duration_frames, comfortably above 48, so
    # every draw should land at or beyond the runway target.
    rng = np.random.default_rng(0)
    for _ in range(20):
        params = sample_synthesis_params(
            PROFILE, "general", clip_frame_count=200, frame_height=100, frame_width=100, fps=24.0, rng=rng
        )
        assert params.window.start_frame >= 48


def test_sample_synthesis_params_squeezes_start_to_latest_when_clip_too_short_for_full_runway():
    # fps=24 -> window_frames=24; PROFILE's max_flashes_per_second=3 ->
    # period_frames = 24 // (2*4) = 3 -> duration_frames = 24 + 12 = 36.
    # clip_frame_count=40 -> latest_start = 40 - 36 = 4, well under the
    # desired_runway of 48 -- the clip cannot offer full runway, so
    # start_frame must be squeezed to exactly latest_start (the only value
    # that still leaves room for the full injection window).
    rng = np.random.default_rng(0)
    for _ in range(20):
        params = sample_synthesis_params(
            PROFILE, "general", clip_frame_count=40, frame_height=100, frame_width=100, fps=24.0, rng=rng
        )
        assert params.window.start_frame == 4
        assert params.window.end_frame == 4 + 36 - 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest training/tests/test_params.py -k "prefers_a_start_with_full_runway or squeezes_start_to_latest" -v`
Expected: FAIL — with the current uniform `rng.integers(1, latest_start + 1)`, `start_frame` can and will land below 48 in the first test (it's drawing from `[1, latest_start]` with no bias), and will not deterministically equal 4 in the second (it draws from `[1, 4]`, not always 4).

- [ ] **Step 3: Implement the placement relaxation**

In `training/params.py`, replace the two lines:

```python
    latest_start = clip_frame_count - duration_frames
    start_frame = int(rng.integers(1, latest_start + 1))
```

with:

```python
    latest_start = clip_frame_count - duration_frames
    # PriorCalc's TargetHistogramSmoother needs a stretch of genuinely clean
    # (non-risky, non-uncertain) frames before the injected segment to ever
    # compute a non-None target histogram. Measured on the real 170-sample
    # dataset: samples whose injected segment started early (median start
    # ~15% into the clip) contributed zero training examples, while samples
    # that started around the clip's middle (median ~48%) contributed dozens
    # each -- see docs/superpowers/specs/2026-08-04-mask-diversification-design.md
    # section 3. When a clip is too short to offer the full desired runway,
    # give it as much as the clip can provide rather than rejecting it
    # outright (ClipTooShortError above already handles clips too short for
    # the injection window itself).
    desired_runway = 2 * window_frames
    min_start = min(desired_runway, latest_start)
    start_frame = int(rng.integers(min_start, latest_start + 1))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest training/tests/test_params.py -v`
Expected: all PASS (including every pre-existing test in this file — `test_sample_synthesis_params_window_fits_inside_clip_and_covers_full_second` and the six shipped-profile parametrized cases in particular, since they all use `clip_frame_count=90, fps=30.0`, giving plenty of room above the new `desired_runway`)

- [ ] **Step 5: Commit**

```bash
git add training/params.py training/tests/test_params.py
git commit -m "fix(dataset-synth): relax injection placement to leave PriorCalc a clean runway"
```

---

## Task 2: Shape rendering primitives

**Files:**
- Create: `training/mask_shapes.py`
- Test: `training/tests/test_mask_shapes.py`

**Interfaces:**
- Consumes: nothing (numpy only — this module knows nothing about `InjectionWindow`, `LightShape`, or DatasetSynth's sampling logic).
- Produces: `rect_area(half_height: int, half_width: int) -> int`, `circle_area(radius: int) -> int`, `beam_area(half_length: int, half_thickness: int) -> int`, `render_rect_mask(frame_height: int, frame_width: int, center_row: float, center_col: float, half_height: int, half_width: int) -> np.ndarray` (bool, shape `(frame_height, frame_width)`), `render_circle_mask(frame_height, frame_width, center_row, center_col, radius) -> np.ndarray`, `render_beam_mask(frame_height, frame_width, center_row, center_col, half_length, half_thickness, angle_degrees) -> np.ndarray`. Consumed by Task 3 (area formulas) and Task 4 (rendering).

- [ ] **Step 1: Write the failing tests**

Create `training/tests/test_mask_shapes.py`:

```python
import numpy as np

from training.mask_shapes import (
    beam_area,
    circle_area,
    rect_area,
    render_beam_mask,
    render_circle_mask,
    render_rect_mask,
)


def test_rect_area_formula():
    assert rect_area(half_height=2, half_width=3) == 5 * 7


def test_render_rect_mask_pixel_count_matches_rect_area():
    mask = render_rect_mask(frame_height=50, frame_width=50, center_row=25, center_col=25, half_height=4, half_width=6)
    assert mask.shape == (50, 50)
    assert mask.dtype == bool
    assert mask.sum() == rect_area(4, 6)


def test_render_rect_mask_is_centered():
    mask = render_rect_mask(frame_height=20, frame_width=20, center_row=10, center_col=10, half_height=2, half_width=2)
    assert mask[10, 10]
    assert mask[8, 10] and mask[12, 10]
    assert not mask[7, 10] and not mask[13, 10]


def test_circle_area_formula():
    assert circle_area(radius=10) == round(np.pi * 100)


def test_render_circle_mask_pixel_count_approximately_matches_circle_area():
    # Pixel rasterization of a circle is never pixel-exact -- allow 15% slack.
    mask = render_circle_mask(frame_height=100, frame_width=100, center_row=50, center_col=50, radius=20)
    expected = circle_area(20)
    assert abs(mask.sum() - expected) / expected < 0.15


def test_render_circle_mask_is_centered_and_round():
    mask = render_circle_mask(frame_height=40, frame_width=40, center_row=20, center_col=20, radius=5)
    assert mask[20, 20]
    assert mask[20, 25] and mask[25, 20]  # exactly on the radius, axis-aligned -- inside
    assert not mask[20, 30]  # well outside
    # Diagonal offset (3, 3): distance = sqrt(18) ~= 4.24 < radius=5 -- inside.
    assert mask[23, 23]
    # Diagonal offset (4, 4): distance = sqrt(32) ~= 5.66 > radius=5 -- outside.
    assert not mask[24, 24]


def test_beam_area_formula():
    assert beam_area(half_length=10, half_thickness=2) == 21 * 5


def test_render_beam_mask_pixel_count_matches_beam_area_at_zero_angle():
    mask = render_beam_mask(
        frame_height=60, frame_width=60, center_row=30, center_col=30,
        half_length=15, half_thickness=3, angle_degrees=0.0,
    )
    assert mask.sum() == beam_area(15, 3)


def test_render_beam_mask_long_axis_follows_angle():
    # At angle=0, the beam is wide along columns (the "length" axis) and
    # narrow along rows (the "thickness" axis).
    mask_0 = render_beam_mask(
        frame_height=60, frame_width=60, center_row=30, center_col=30,
        half_length=20, half_thickness=2, angle_degrees=0.0,
    )
    assert mask_0[30, 45]       # far along columns from center -- inside
    assert not mask_0[45, 30]   # far along rows from center -- outside

    # At angle=90, the beam's long axis rotates to follow rows instead.
    mask_90 = render_beam_mask(
        frame_height=60, frame_width=60, center_row=30, center_col=30,
        half_length=20, half_thickness=2, angle_degrees=90.0,
    )
    assert mask_90[45, 30]       # far along rows -- now inside
    assert not mask_90[30, 45]   # far along columns -- now outside


def test_render_masks_stay_within_frame_bounds_near_an_edge():
    # A shape centered near/at a corner must not raise and must return a
    # full-sized array -- pixels beyond the frame are simply not covered.
    for render, kwargs in [
        (render_rect_mask, dict(half_height=5, half_width=5)),
        (render_circle_mask, dict(radius=5)),
        (render_beam_mask, dict(half_length=10, half_thickness=3, angle_degrees=45.0)),
    ]:
        mask = render(frame_height=20, frame_width=20, center_row=0, center_col=0, **kwargs)
        assert mask.shape == (20, 20)
        assert mask.any()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest training/tests/test_mask_shapes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.mask_shapes'`

- [ ] **Step 3: Implement `training/mask_shapes.py`**

```python
"""Renders boolean pixel masks for the shapes DatasetSynth's realistic
injection path can move around a frame (rectangle, circle, beam), plus the
analytic pixel area each shape occupies.

Kept independent of InjectionWindow/LightShape (training.params) and of
color logic (training.injection) so shape geometry has exactly one place to
live and one thing to test -- neither of those modules' tests need to
reason about rasterization, and this module's tests never need a
ThresholdProfile or an RNG.
"""
import numpy as np


def rect_area(half_height: int, half_width: int) -> int:
    return (2 * half_height + 1) * (2 * half_width + 1)


def circle_area(radius: int) -> int:
    return round(np.pi * radius ** 2)


def beam_area(half_length: int, half_thickness: int) -> int:
    return (2 * half_length + 1) * (2 * half_thickness + 1)


def render_rect_mask(
    frame_height: int,
    frame_width: int,
    center_row: float,
    center_col: float,
    half_height: int,
    half_width: int,
) -> np.ndarray:
    rows = np.arange(frame_height)[:, None].astype(np.float64)
    cols = np.arange(frame_width)[None, :].astype(np.float64)
    return (np.abs(rows - center_row) <= half_height) & (np.abs(cols - center_col) <= half_width)


def render_circle_mask(
    frame_height: int,
    frame_width: int,
    center_row: float,
    center_col: float,
    radius: int,
) -> np.ndarray:
    rows = np.arange(frame_height)[:, None].astype(np.float64)
    cols = np.arange(frame_width)[None, :].astype(np.float64)
    return (rows - center_row) ** 2 + (cols - center_col) ** 2 <= radius ** 2


def render_beam_mask(
    frame_height: int,
    frame_width: int,
    center_row: float,
    center_col: float,
    half_length: int,
    half_thickness: int,
    angle_degrees: float,
) -> np.ndarray:
    rows = np.arange(frame_height)[:, None].astype(np.float64) - center_row
    cols = np.arange(frame_width)[None, :].astype(np.float64) - center_col
    theta = np.deg2rad(angle_degrees)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Rotate frame coordinates into the beam's own axes so the beam is an
    # axis-aligned rectangle (length along local_long, thickness along
    # local_short) in its own frame -- same trick as rotating a bounding box.
    local_long = cols * cos_t + rows * sin_t
    local_short = -cols * sin_t + rows * cos_t
    return (np.abs(local_long) <= half_length) & (np.abs(local_short) <= half_thickness)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest training/tests/test_mask_shapes.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add training/mask_shapes.py training/tests/test_mask_shapes.py
git commit -m "feat(dataset-synth): add rect/circle/beam mask rendering primitives"
```

---

## Task 3: `LightShape` + `sample_lights`

**Files:**
- Modify: `training/params.py` (add `LightShape`, extend `InjectionWindow`, add `sample_lights`)
- Test: `training/tests/test_params.py`

**Interfaces:**
- Consumes: `training.mask_shapes.{rect_area, circle_area, beam_area}` (Task 2).
- Produces: `LightShape` dataclass (fields: `kind: str`, `start_row: float`, `start_col: float`, `end_row: float`, `end_col: float`, `half_height: int | None`, `half_width: int | None`, `radius: int | None`, `half_length: int | None`, `half_thickness: int | None`, `angle_degrees: float | None`). `InjectionWindow` gains `lights: list[LightShape] = field(default_factory=list)`. `sample_lights(profile: ThresholdProfile, frame_height: int, frame_width: int, rng: np.random.Generator) -> list[LightShape]`. Consumed by Task 4 (rendering) and Task 5 (`sample_synthesis_params_realistic`).

- [ ] **Step 1: Write the failing tests**

Add to `training/tests/test_params.py` (add `from training import mask_shapes` and `from training.params import LightShape, sample_lights` to the existing imports at the top):

```python
def test_sample_lights_returns_one_to_three_lights():
    rng = np.random.default_rng(0)
    counts = {len(sample_lights(PROFILE, 100, 100, rng)) for _ in range(30)}
    assert counts <= {1, 2, 3}
    assert counts  # at least one draw happened


def test_sample_lights_kinds_are_valid():
    rng = np.random.default_rng(0)
    for _ in range(10):
        for light in sample_lights(PROFILE, 100, 100, rng):
            assert light.kind in ("rect", "circle", "beam")


def test_sample_lights_combined_area_exceeds_profile_ratio():
    def area_of(light):
        if light.kind == "rect":
            return mask_shapes.rect_area(light.half_height, light.half_width)
        if light.kind == "circle":
            return mask_shapes.circle_area(light.radius)
        return mask_shapes.beam_area(light.half_length, light.half_thickness)

    rng = np.random.default_rng(0)
    for _ in range(20):
        lights = sample_lights(PROFILE, 100, 100, rng)
        total_area = sum(area_of(light) for light in lights)
        assert total_area / (100 * 100) > PROFILE.max_area_ratio


def test_sample_lights_positions_are_within_frame_bounds():
    rng = np.random.default_rng(0)
    for _ in range(20):
        for light in sample_lights(PROFILE, 80, 60, rng):
            assert 0 <= light.start_row <= 80 and 0 <= light.end_row <= 80
            assert 0 <= light.start_col <= 60 and 0 <= light.end_col <= 60


def test_sample_lights_rejects_profile_whose_area_target_is_unreachable():
    profile = ThresholdProfile(
        name="area-cliff", max_flashes_per_second=3, max_area_ratio=0.95,
        general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
        red_saturation_ratio_threshold=0.80,
    )
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="area-cliff"):
        sample_lights(profile, 100, 100, rng)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest training/tests/test_params.py -k sample_lights -v`
Expected: FAIL with `ImportError: cannot import name 'sample_lights'` (and `'LightShape'`)

- [ ] **Step 3: Implement `LightShape`, `InjectionWindow.lights`, and `sample_lights`**

In `training/params.py`:

Change the import line:
```python
from dataclasses import dataclass
```
to:
```python
from dataclasses import dataclass, field
```

Add near the top of the file (module-level import section):
```python
from training import mask_shapes
```

Add a new dataclass right after `ClipTooShortError` and before `InjectionWindow`:
```python
@dataclass
class LightShape:
    kind: str  # "rect" | "circle" | "beam"
    start_row: float
    start_col: float
    end_row: float
    end_col: float
    half_height: int | None = None    # rect only
    half_width: int | None = None     # rect only
    radius: int | None = None         # circle only
    half_length: int | None = None    # beam only
    half_thickness: int | None = None  # beam only
    angle_degrees: float | None = None  # beam only
```

Change `InjectionWindow` to add one field:
```python
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
    # Populated only by sample_synthesis_params_realistic (Task 5) -- empty
    # for the flat-mode path and any window built by plain
    # sample_synthesis_params, which keeps using the mask_top/left/height/
    # width rectangle above via injection.py's fallback path.
    lights: list[LightShape] = field(default_factory=list)
```

Add near the bottom of the file (after `sample_synthesis_params`):
```python
_LIGHT_KINDS = ("rect", "circle", "beam")
_MAX_LIGHTS = 3
_BEAM_ASPECT = 4  # half_length = _BEAM_ASPECT * half_thickness
# Deriving a shape's half-extent/radius from a target area via round() can
# overshoot the target by several percent per light (squaring amplifies the
# rounding error), and summing that overshoot across up to _MAX_LIGHTS lights
# compounds it further -- empirically, two rect lights each targeting 4500px^2
# out of a 10000px^2 frame achieved a combined 9522px^2 (95.2%), which would
# have silently defeated _require_exceeds for a profile whose max_area_ratio
# sits at 0.95 (right where _MAX_AREA_RATIO would otherwise cap it). Capping
# the combined *lights* target well below _MAX_AREA_RATIO leaves enough
# headroom to absorb that per-light overshoot; the legacy single-rectangle
# path (_mask_dims) is unaffected and keeps using _MAX_AREA_RATIO directly.
_MAX_LIGHTS_AREA_RATIO = 0.60


def _light_area(light: LightShape) -> int:
    if light.kind == "rect":
        return mask_shapes.rect_area(light.half_height, light.half_width)
    if light.kind == "circle":
        return mask_shapes.circle_area(light.radius)
    return mask_shapes.beam_area(light.half_length, light.half_thickness)


def _sample_one_light(
    kind: str,
    target_area: float,
    frame_height: int,
    frame_width: int,
    rng: np.random.Generator,
) -> LightShape:
    start_row, start_col = float(rng.uniform(0, frame_height)), float(rng.uniform(0, frame_width))
    end_row, end_col = float(rng.uniform(0, frame_height)), float(rng.uniform(0, frame_width))
    max_half = max(1, min(frame_height, frame_width) // 2)

    if kind == "rect":
        half = max(1, min(round((target_area ** 0.5) / 2), max_half))
        return LightShape(
            kind="rect", start_row=start_row, start_col=start_col,
            end_row=end_row, end_col=end_col, half_height=half, half_width=half,
        )
    if kind == "circle":
        radius = max(1, min(round((target_area / np.pi) ** 0.5), max_half))
        return LightShape(
            kind="circle", start_row=start_row, start_col=start_col,
            end_row=end_row, end_col=end_col, radius=radius,
        )
    half_thickness = max(1, min(round((target_area / (4 * _BEAM_ASPECT)) ** 0.5), max_half))
    half_length = max(1, min(_BEAM_ASPECT * half_thickness, max(frame_height, frame_width) // 2))
    angle_degrees = float(rng.uniform(0, 180))
    return LightShape(
        kind="beam", start_row=start_row, start_col=start_col,
        end_row=end_row, end_col=end_col, half_length=half_length,
        half_thickness=half_thickness, angle_degrees=angle_degrees,
    )


def sample_lights(
    profile: ThresholdProfile,
    frame_height: int,
    frame_width: int,
    rng: np.random.Generator,
) -> list[LightShape]:
    n_lights = int(rng.integers(1, _MAX_LIGHTS + 1))
    target_total_area = min(
        (profile.max_area_ratio + _AREA_MARGIN) * frame_height * frame_width,
        _MAX_LIGHTS_AREA_RATIO * frame_height * frame_width,
    )
    per_light_area = target_total_area / n_lights

    lights = [
        _sample_one_light(
            _LIGHT_KINDS[int(rng.integers(0, len(_LIGHT_KINDS)))],
            per_light_area, frame_height, frame_width, rng,
        )
        for _ in range(n_lights)
    ]

    achieved_area_ratio = sum(_light_area(light) for light in lights) / (frame_height * frame_width)
    _require_exceeds(
        achieved_area_ratio, profile.max_area_ratio, profile.name,
        "max_area_ratio", "combined injected light area ratio",
        f"light sizes are derived from an equal per-light share of "
        f"{_MAX_LIGHTS_AREA_RATIO:.2f} of the frame (_MAX_LIGHTS_AREA_RATIO) "
        f"and rounded to whole pixels on {frame_height}x{frame_width}",
    )
    return lights
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest training/tests/test_params.py -v`
Expected: all PASS, including every pre-existing test in the file (the new `lights` field defaults to `[]` via `field(default_factory=list)`, so every existing `InjectionWindow(...)` construction that doesn't pass `lights=` keeps working identically).

- [ ] **Step 5: Commit**

```bash
git add training/params.py training/tests/test_params.py
git commit -m "feat(dataset-synth): add LightShape and sample_lights"
```

---

## Task 4: Move lights per-frame inside the realistic injection functions

**Files:**
- Modify: `training/injection.py`
- Test: `training/tests/test_injection.py`

**Interfaces:**
- Consumes: `training.params.LightShape` (Task 3), `training.mask_shapes.{render_rect_mask, render_circle_mask, render_beam_mask}` (Task 2).
- Produces: `inject_general_flash_realistic`/`inject_red_flash_realistic` keep their existing signatures; when `window.lights` is non-empty they render a per-frame union-of-lights mask instead of the static rectangle. `inject_general_flash`/`inject_red_flash` (flat mode) are untouched.

- [ ] **Step 1: Write the failing tests**

Add to `training/tests/test_injection.py` (add `from training.params import InjectionWindow, LightShape` — replace the existing `from training.params import InjectionWindow` line):

```python
def test_inject_general_flash_realistic_with_moving_light_hits_start_then_end_position():
    frames = _solid_frames(11, 30, 30, 0.5)
    light = LightShape(kind="circle", start_row=5, start_col=5, end_row=25, end_col=25, radius=2)
    window = InjectionWindow(
        start_frame=0, end_frame=10, mask_top=0, mask_left=0, mask_height=1, mask_width=1,
        period_frames=1, ramp_frames=1, lights=[light],
    )
    out = inject_general_flash_realistic(frames, window, gain_dark=0.3, gain_bright=3.0)
    # At start_frame the light sits near (5,5); its gain (dark, offset 0 is
    # not a pulse frame per _is_pulse_frame) must have changed that pixel...
    assert out[0][5, 5, 0] != frames[0][5, 5, 0]
    assert out[0][25, 25, 0] == frames[0][25, 25, 0]  # end position untouched yet
    # ...and at end_frame the light has moved to (25,25).
    assert out[10][25, 25, 0] != frames[10][25, 25, 0]
    assert out[10][5, 5, 0] == frames[10][5, 5, 0]  # start position no longer lit


def test_inject_general_flash_realistic_with_two_lights_unions_both_regions():
    frames = _solid_frames(4, 30, 30, 0.5)
    light_a = LightShape(kind="circle", start_row=5, start_col=5, end_row=5, end_col=5, radius=2)
    light_b = LightShape(kind="circle", start_row=25, start_col=25, end_row=25, end_col=25, radius=2)
    window = InjectionWindow(
        start_frame=0, end_frame=3, mask_top=0, mask_left=0, mask_height=1, mask_width=1,
        period_frames=1, ramp_frames=1, lights=[light_a, light_b],
    )
    out = inject_general_flash_realistic(frames, window, gain_dark=0.3, gain_bright=3.0)
    assert out[0][5, 5, 0] != frames[0][5, 5, 0]
    assert out[0][25, 25, 0] != frames[0][25, 25, 0]
    assert out[0][15, 15, 0] == frames[0][15, 15, 0]  # between the two lights: untouched


def test_inject_red_flash_realistic_with_moving_light_hits_start_then_end_position():
    frames = _solid_frames(11, 30, 30, 0.3)
    light = LightShape(kind="circle", start_row=5, start_col=5, end_row=25, end_col=25, radius=2)
    window = InjectionWindow(
        start_frame=0, end_frame=10, mask_top=0, mask_left=0, mask_height=1, mask_width=1,
        period_frames=1, ramp_frames=1, lights=[light],
    )
    out = inject_red_flash_realistic(frames, window)
    assert not np.array_equal(out[0][5, 5], frames[0][5, 5])
    assert np.array_equal(out[0][25, 25], frames[0][25, 25])
    assert not np.array_equal(out[10][25, 25], frames[10][25, 25])
    assert np.array_equal(out[10][5, 5], frames[10][5, 5])


def test_inject_general_flash_realistic_lights_leave_frames_outside_window_untouched():
    frames = _solid_frames(10, 20, 20, 0.5)
    light = LightShape(kind="rect", start_row=10, start_col=10, end_row=10, end_col=10, half_height=8, half_width=8)
    window = InjectionWindow(
        start_frame=3, end_frame=6, mask_top=0, mask_left=0, mask_height=1, mask_width=1,
        period_frames=1, ramp_frames=1, lights=[light],
    )
    out = inject_general_flash_realistic(frames, window)
    assert np.array_equal(out[0], frames[0])
    assert np.array_equal(out[9], frames[9])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest training/tests/test_injection.py -k moving_light or two_lights -v`
Expected: FAIL — the current implementation always uses `_mask_slice(window)` (the legacy `mask_top/left/height/width` rectangle, which these test windows deliberately set to a 1x1 no-op region), so none of the light-shaped regions would actually change.

- [ ] **Step 3: Implement per-frame light rendering**

In `training/injection.py`, add the import (extend the existing `from training.params import InjectionWindow` line):
```python
from training.params import InjectionWindow, LightShape
```
and:
```python
from training import mask_shapes
```

Add after `_mask_slice`:
```python
def _interpolate(start: float, end: float, offset: int, span: int) -> float:
    if span <= 0:
        return start
    t = offset / span
    return start + (end - start) * t


def _light_mask_at_offset(
    light: LightShape, frame_height: int, frame_width: int, offset: int, span: int,
) -> np.ndarray:
    row = _interpolate(light.start_row, light.end_row, offset, span)
    col = _interpolate(light.start_col, light.end_col, offset, span)
    if light.kind == "rect":
        return mask_shapes.render_rect_mask(frame_height, frame_width, row, col, light.half_height, light.half_width)
    if light.kind == "circle":
        return mask_shapes.render_circle_mask(frame_height, frame_width, row, col, light.radius)
    return mask_shapes.render_beam_mask(
        frame_height, frame_width, row, col, light.half_length, light.half_thickness, light.angle_degrees
    )


def _region_mask_at_offset(
    window: InjectionWindow, frame_height: int, frame_width: int, offset: int,
) -> np.ndarray:
    if not window.lights:
        rows, cols = _mask_slice(window)
        mask = np.zeros((frame_height, frame_width), dtype=bool)
        mask[rows, cols] = True
        return mask
    span = window.end_frame - window.start_frame
    mask = np.zeros((frame_height, frame_width), dtype=bool)
    for light in window.lights:
        mask |= _light_mask_at_offset(light, frame_height, frame_width, offset, span)
    return mask
```

Replace `inject_general_flash_realistic`'s body:
```python
def inject_general_flash_realistic(
    frames: list[np.ndarray],
    window: InjectionWindow,
    gain_dark: float = 0.3,
    gain_bright: float = 3.0,
) -> list[np.ndarray]:
    """Alternates the injected region between two *multiplicative* luminance
    gains in linear light space, instead of inject_general_flash's flat
    color overwrite -- this preserves the underlying scene's texture, matching
    how a real light source's intensity change multiplies reflected light
    physically. Gain defaults validated empirically (see plan/spec) to
    reliably cross Detector's default thresholds on real footage. The
    injected region is window.lights's per-frame union when set (moving
    shapes), or the legacy static rectangle when not (see
    _region_mask_at_offset)."""
    frame_height, frame_width = frames[0].shape[:2]
    out = [frame.copy() for frame in frames]
    for i in range(window.start_frame, window.end_frame + 1):
        offset = i - window.start_frame
        gain = gain_bright if _is_pulse_frame(offset, window.period_frames) else gain_dark
        mask = _region_mask_at_offset(window, frame_height, frame_width, offset)
        region = out[i][mask, :]
        linear = _srgb_to_linear(region)
        out[i][mask, :] = _linear_to_srgb(linear * gain)
    return out
```

Replace `inject_red_flash_realistic`'s body:
```python
def inject_red_flash_realistic(
    frames: list[np.ndarray],
    window: InjectionWindow,
    red_gains: tuple[float, float, float] = (20.0, 0.02, 0.02),
    baseline_gains: tuple[float, float, float] = (0.3, 1.0, 1.0),
) -> list[np.ndarray]:
    """Alternates the injected region between two per-channel gain triples in
    linear light space -- red_gains boosts R and suppresses G/B, baseline_gains
    suppresses R -- instead of inject_red_flash's flat saturated-color
    overwrite. Both states are deliberately engineered (neither is a pass-
    through of the original), because a pass-through baseline can itself
    exceed the saturation-ratio threshold on already-reddish content,
    breaking the XOR-based transition Detector relies on. gain magnitudes are
    large because the WCAG saturation-ratio test runs in gamma-compressed
    sRGB space, which compresses a 10x linear channel-ratio difference down
    to roughly 2.6x -- verified empirically (see plan/spec). The injected
    region is window.lights's per-frame union when set (moving shapes), or
    the legacy static rectangle when not (see _region_mask_at_offset)."""
    frame_height, frame_width = frames[0].shape[:2]
    out = [frame.copy() for frame in frames]
    for i in range(window.start_frame, window.end_frame + 1):
        offset = i - window.start_frame
        gains = red_gains if _is_pulse_frame(offset, window.period_frames) else baseline_gains
        mask = _region_mask_at_offset(window, frame_height, frame_width, offset)
        region = out[i][mask, :]
        linear = _srgb_to_linear(region)
        modulated = linear * np.array(gains, dtype=np.float32)
        out[i][mask, :] = _linear_to_srgb(modulated)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest training/tests/test_injection.py -v`
Expected: all PASS, including every pre-existing test in the file (they never set `lights`, so `_region_mask_at_offset` takes the `not window.lights` branch and reproduces the exact old `_mask_slice`-based rectangle behavior).

- [ ] **Step 5: Commit**

```bash
git add training/injection.py training/tests/test_injection.py
git commit -m "feat(dataset-synth): render moving multi-shape lights in realistic injection"
```

---

## Task 5: Wire lights into `synthesize_sample_realistic`

**Files:**
- Modify: `training/params.py` (add `sample_synthesis_params_realistic`)
- Modify: `training/synth.py:78-116` (`_apply_pattern_realistic` no change needed; `synthesize_sample_realistic` swaps which sampler it calls)
- Test: `training/tests/test_params.py`, `training/tests/test_synth.py`, `training/tests/test_dataset_writer.py`

**Interfaces:**
- Consumes: `sample_synthesis_params` (unchanged signature, Task 1), `sample_lights` (Task 3).
- Produces: `sample_synthesis_params_realistic(profile, pattern, clip_frame_count, frame_height, frame_width, fps, rng) -> SynthParams` — same signature and return type as `sample_synthesis_params`, but `result.window.lights` is populated. `synthesize_sample_realistic`'s public signature is unchanged.

- [ ] **Step 1: Write the failing tests**

Add to `training/tests/test_params.py`:
```python
def test_sample_synthesis_params_realistic_populates_lights():
    rng = np.random.default_rng(0)
    params = sample_synthesis_params_realistic(
        PROFILE, "general", clip_frame_count=200, frame_height=100, frame_width=100, fps=24.0, rng=rng
    )
    assert 1 <= len(params.window.lights) <= 3


def test_sample_synthesis_params_realistic_keeps_the_same_placement_and_timing_logic():
    # Same runway relaxation as plain sample_synthesis_params (Task 1) --
    # this wrapper must not re-derive start_frame/end_frame independently.
    rng = np.random.default_rng(0)
    for _ in range(20):
        params = sample_synthesis_params_realistic(
            PROFILE, "general", clip_frame_count=200, frame_height=100, frame_width=100, fps=24.0, rng=rng
        )
        assert params.window.start_frame >= 48
```

Add `sample_synthesis_params_realistic` to the existing `from training.params import ...` line in `training/tests/test_params.py`.

In `training/tests/test_synth.py`, first REPLACE the existing test that reads the legacy rectangle fields (it becomes structurally invalid once `synthesize_sample_realistic` draws lights instead of relying on `mask_top/left/height/width` for the actual injected pixels):

```python
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
```

Then add:
```python
def test_synthesize_sample_realistic_accepts_general_flash_with_lights():
    rng = np.random.default_rng(0)
    sample = synthesize_sample_realistic(_textured_clip(), fps=30.0, profile=PROFILE, pattern="general", rng=rng)
    assert sample is not None
    assert 1 <= len(sample.window.lights) <= 3
```

In `training/tests/test_dataset_writer.py`, add:
```python
def test_write_sample_round_trips_lights_through_meta_json(tmp_path):
    from training.params import LightShape

    sample = _sample()
    sample.window.lights = [
        LightShape(kind="circle", start_row=1.0, start_col=2.0, end_row=3.0, end_col=4.0, radius=2),
    ]
    sid = write_sample(sample, tmp_path, clip_id="bear", index=0, injection_mode="realistic", fps=24.0)
    meta = json.loads((tmp_path / sid / "meta.json").read_text(encoding="utf-8"))
    lights = meta["injected_window"]["lights"]
    assert lights == [
        {
            "kind": "circle", "start_row": 1.0, "start_col": 2.0, "end_row": 3.0, "end_col": 4.0,
            "half_height": None, "half_width": None, "radius": 2,
            "half_length": None, "half_thickness": None, "angle_degrees": None,
        }
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest training/tests/test_params.py -k realistic_populates_lights -v training/tests/test_synth.py -k lights -v training/tests/test_dataset_writer.py -k round_trips_lights -v`
Expected: FAIL with `ImportError`/`AttributeError` (`sample_synthesis_params_realistic` doesn't exist yet; `synthesize_sample_realistic` doesn't populate lights yet).

Run also: `pytest training/tests/test_synth.py -k preserves_spatial_texture -v`
Expected: PASS already (the rewritten version doesn't depend on anything new) — confirms the rewrite is valid against the *current* code before Step 3 changes behavior.

- [ ] **Step 3: Implement `sample_synthesis_params_realistic` and wire it into `synth.py`**

In `training/params.py`, add after `sample_lights`:
```python
def sample_synthesis_params_realistic(
    profile: ThresholdProfile,
    pattern: str,
    clip_frame_count: int,
    frame_height: int,
    frame_width: int,
    fps: float,
    rng: np.random.Generator,
) -> SynthParams:
    """Same placement/timing as sample_synthesis_params, but replaces the
    single static rectangle with a set of moving lights (see
    docs/superpowers/specs/2026-08-04-mask-diversification-design.md) --
    used only by the realistic injection path (synthesize_sample_realistic).
    The flat path keeps calling sample_synthesis_params directly, so its
    single static rectangle is unaffected."""
    params = sample_synthesis_params(profile, pattern, clip_frame_count, frame_height, frame_width, fps, rng)
    params.window.lights = sample_lights(profile, frame_height, frame_width, rng)
    return params
```

In `training/synth.py`, change the import line:
```python
from training.params import InjectionWindow, SynthParams, sample_synthesis_params
```
to:
```python
from training.params import InjectionWindow, SynthParams, sample_synthesis_params, sample_synthesis_params_realistic
```

In `synthesize_sample_realistic`, change:
```python
        params = sample_synthesis_params(
            profile, pattern, len(clean_frames), frame_height, frame_width, fps, rng
        )
```
to:
```python
        params = sample_synthesis_params_realistic(
            profile, pattern, len(clean_frames), frame_height, frame_width, fps, rng
        )
```
(this is the only line inside `synthesize_sample_realistic` that changes — everything else in that function, and all of `synthesize_sample`, stays exactly as-is).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest training/tests/ -v`
Expected: all PASS across the whole `training/tests/` suite. Pay particular attention to `test_synthesize_sample_realistic_accepts_general_flash`, `test_synthesize_sample_realistic_accepts_red_flash`, and the new `..._with_lights` test — these depend on the fixed seeds (`rng=0`, `rng=1`) still producing a light configuration that `run_detection` flags as risky within `max_retries=5` attempts. If any of these fail (returns `None`), that is real signal, not a fluke: try the next few integer seeds in the test itself (e.g. `np.random.default_rng(2)`) and use the first one that passes reliably across a few repeat runs — do not increase `max_retries` or loosen `_window_is_covered` to force it, since that would mask a real area/coverage problem with the light-sampling logic instead of fixing it.

- [ ] **Step 5: Commit**

```bash
git add training/params.py training/synth.py training/tests/test_params.py training/tests/test_synth.py training/tests/test_dataset_writer.py
git commit -m "feat(dataset-synth): wire moving multi-shape lights into synthesize_sample_realistic"
```

---

## Task 6: Package verification + real-data regeneration + yield re-measurement

**Files:**
- No new source files.
- Modify: `docs/superpowers/specs/2026-08-03-mitigator-design.md` §8 (update the yield note with fresh numbers)

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: nothing new — this task is verification plus a real re-synthesis of `data/synthetic/`.

- [ ] **Step 1: Run the full repository test suite**

Run: `pytest -v` from the repo root.
Expected: all tests pass (should be the pre-existing 182 plus this plan's new tests — count them from the `-v` output and record the total in your report/commit message).

- [ ] **Step 2: Manual verification (not an automated test) — real 90-clip regeneration**

This step is a manual, one-time check, matching this project's established pattern (Detector's/DatasetSynth's/Mitigator's own plans all had one). From the repo root (use the real clip/profile directories, not the small unit-test fixtures):

```bash
python -m training.cli --clips-dir data/clips_all --profiles-dir data/profiles_active --output data/synthetic --injection-mode realistic --overwrite
```

Expected: completes without error, `data/synthetic/summary.json`'s `accepted` count is in the same ballpark as before (170/180, 94.4%) -- this plan does not change which clips get accepted (`ClipTooShortError`/`validation_exhausted` triggers are unchanged), only where each accepted sample's target histogram becomes computable.

- [ ] **Step 3: Manual verification — measure the actual yield improvement**

Run this from the repo root against the freshly regenerated `data/synthetic/`:

```bash
python -c "
import json
from pathlib import Path
import numpy as np

n_zero, n_nonzero, total_examples = 0, 0, 0
clips_with_examples = set()
for sample_dir in sorted(Path('data/synthetic').iterdir()):
    meta_path = sample_dir / 'meta.json'
    if not meta_path.exists():
        continue
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    cache = np.load(sample_dir / 'prior_cache.npz') if (sample_dir / 'prior_cache.npz').exists() else None
    # prior_cache.npz may not exist yet on a fresh regen -- if so, this
    # script's caller should first construct a MitigatorDataset for both
    # splits (as training/train_mitigator.py's main() already does) to
    # populate it, then rerun this measurement.
    if cache is None:
        continue
    has_target = cache['has_target']
    frame_indices = cache['frame_indices']
    idx_of = {int(fi): i for i, fi in enumerate(frame_indices)}
    risky = set()
    for s in meta['segments']:
        risky.update(range(s['start_frame'], s['end_frame'] + 1))
    contributing = sum(1 for fi in risky if fi in idx_of and has_target[idx_of[fi]])
    total_examples += contributing
    if contributing > 0:
        n_nonzero += 1
        clips_with_examples.add(meta['clip_id'])
    else:
        n_zero += 1

print(f'zero-yield samples: {n_zero}, contributing samples: {n_nonzero}')
print(f'total contributing examples: {total_examples}, distinct clips: {len(clips_with_examples)}')
"
```

If `prior_cache.npz` files are missing (fresh regen), first run the training-side dataset construction once to populate them (mirrors `training/train_mitigator.py main()`'s own dataset construction):
```bash
python -c "
from pathlib import Path
from detector.profiles import load_profile
from training.mitigator_dataset import MitigatorDataset
profile = load_profile(Path('configs/profiles/itu.json'))
MitigatorDataset(Path('data/synthetic'), profile, split='train')
MitigatorDataset(Path('data/synthetic'), profile, split='val')
"
```

Expected: `n_nonzero` and `len(clips_with_examples)` are meaningfully higher than the pre-fix baseline (17 contributing samples / 15 distinct clips, out of 170). Record the actual before/after numbers.

- [ ] **Step 4: Update the design doc with real numbers**

Edit `docs/superpowers/specs/2026-08-03-mitigator-design.md` §8's `실측된 실제 수율` bullet: replace the old 153/170 (90%) / 17-samples-15-clips figures with the numbers measured in Step 3, and note that this was after the placement relaxation + mask diversification (`docs/superpowers/specs/2026-08-04-mask-diversification-design.md`) shipped.

- [ ] **Step 5: Final commit**

```bash
git add docs/superpowers/specs/2026-08-03-mitigator-design.md
git status --short
```

If `data/synthetic/` or `mitigator/weights/*.pt` changed on disk, do NOT commit them (gitignored, regenerable, matches this project's existing convention). Commit only the design doc update:

```bash
git commit -m "docs(mitigator): record post-fix injection yield measurement"
```
