# Synthesis Parameter Diversification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace DatasetSynth's fixed-formula mask area and brightness/color contrast with randomized sampling, so training data spans a realistic severity range instead of clustering at "just barely over the profile threshold".

**Architecture:** All changes are confined to sampling logic in `training/params.py`, plus one defensive `except` clause in `training/cli.py`'s batch loop. No changes to injection rendering, the Detector re-validation loop, or any data format — a diversified sample is structurally identical to a current one, only its numeric parameters differ. Existing `_require_exceeds` validation stays in place on every path, so no sampled value can fall below its profile threshold.

**Tech Stack:** Python 3.10+, numpy (`np.random.Generator` for all sampling). No new dependencies.

## Global Constraints

- Every randomized range must be defined so the value the current fixed formula produces lies **inside** the new range (at one end). Diversification only widens; it never excludes what already worked.
- `_require_exceeds` calls stay on every path they currently guard. Randomization must never let a sample fall below the profile's threshold.
- All randomness goes through the `rng: np.random.Generator` already threaded into these functions. Never call `np.random.*` module-level functions — per-sample reproducibility (`training/cli.py`'s `_sample_rng(seed, sid)`) depends on it.
- Follow the existing file's comment style: comments explain *why* a value or bound was chosen, never *what* the code does.
- Out of scope, do not implement: risky-segment duration diversification, multiple risky segments per clip, source-clip domain swap, MitigatorNet architecture changes.

---

## File Structure

```
flicker-guard/
├── training/
│   ├── params.py          # Tasks 1, 2, 3 (modify) — all sampling logic
│   ├── cli.py             # Task 4 (modify) — batch-loop ValueError guard
│   └── tests/
│       ├── test_params.py # Tasks 1, 2, 3 (modify) — sampling tests
│       └── test_cli.py    # Task 4 (modify) — batch guard test
```

Task ordering rationale: Task 1 (shape-kind filtering) must land before Task 2 (area randomization), because Task 2 raises area targets into exactly the range where Task 1's fix is required to avoid a ~33% `ValueError` rate. Task 3 (contrast) is independent of both. Task 4 (batch guard) is defense-in-depth and independent.

---

## Task 1: Exclude shape kinds that cannot reach the target area

**Files:**
- Modify: `training/params.py` (add `_max_achievable_area`, `_feasible_kinds`; modify `sample_lights`'s kind selection)
- Test: `training/tests/test_params.py`

**Interfaces:**
- Consumes: existing `mask_shapes.rect_area(half_height, half_width) -> int`, `mask_shapes.circle_area(radius) -> int`, `mask_shapes.beam_area(half_length, half_thickness) -> int`; existing module constants `_LIGHT_KINDS = ("rect", "circle", "beam")`, `_BEAM_ASPECT = 4`.
- Produces: `_max_achievable_area(kind: str, frame_height: int, frame_width: int) -> int` and `_feasible_kinds(target_area: float, frame_height: int, frame_width: int) -> tuple[str, ...]`, both consumed by `sample_lights` in this task and unchanged by later tasks.

**Background for the implementer:** `_sample_one_light` clamps every shape's size to at most half the frame (`max_half`), so each kind has a hard ceiling on the area it can produce. Measured on a 480x854 frame: beam tops out at ratio ~0.52, circle at ~0.44, rect at ~0.56. When `sample_lights` picks a kind uniformly and that kind cannot reach its share of the target area, the combined-area `_require_exceeds` check fails and raises a plain `ValueError` — measured at ~33% of draws when the target ratio is 0.60. Task 2 raises targets into exactly that range, so this must be fixed first.

- [ ] **Step 1: Write the failing tests**

Add to `training/tests/test_params.py`:

```python
def test_max_achievable_area_matches_what_sample_one_light_actually_produces():
    # _feasible_kinds' filtering is only correct if _max_achievable_area
    # agrees with the real clamps inside _sample_one_light -- if they drift
    # apart, filtering either lets through a kind that still fails or
    # needlessly excludes one that would have worked.
    rng = np.random.default_rng(0)
    frame_height, frame_width = 480, 854
    huge_target = float(frame_height * frame_width)  # forces every clamp to bind

    for kind in params._LIGHT_KINDS:
        light = params._sample_one_light(kind, huge_target, frame_height, frame_width, rng)
        assert params._light_area(light) == params._max_achievable_area(kind, frame_height, frame_width)


def test_feasible_kinds_keeps_all_three_for_small_targets():
    # A small target is reachable by every shape -- filtering must not
    # collapse shape variety in the common case.
    frame_height, frame_width = 480, 854
    small_target = 0.05 * frame_height * frame_width
    assert set(params._feasible_kinds(small_target, frame_height, frame_width)) == set(params._LIGHT_KINDS)


def test_feasible_kinds_drops_kinds_that_cannot_reach_a_large_target():
    # Measured on 480x854: beam caps near ratio 0.52 and circle near 0.44,
    # while rect reaches ~0.56 -- so a 0.50 target must exclude circle, and
    # only rect can serve a 0.55 target.
    frame_height, frame_width = 480, 854

    kinds_at_050 = params._feasible_kinds(0.50 * frame_height * frame_width, frame_height, frame_width)
    assert "rect" in kinds_at_050
    assert "circle" not in kinds_at_050

    kinds_at_055 = params._feasible_kinds(0.55 * frame_height * frame_width, frame_height, frame_width)
    assert kinds_at_055 == ("rect",)


def test_feasible_kinds_never_returns_empty():
    # sample_lights picks uniformly from this tuple -- an empty result would
    # be an IndexError instead of the clear _require_exceeds ValueError that
    # is supposed to report an unsatisfiable profile.
    frame_height, frame_width = 480, 854
    impossible_target = float(frame_height * frame_width) * 10
    assert params._feasible_kinds(impossible_target, frame_height, frame_width) != ()


def test_sample_lights_does_not_raise_at_high_area_targets():
    # Regression test: with uniform kind selection this raised ValueError on
    # roughly a third of draws at a 0.60 target (measured), and because it is
    # a plain ValueError rather than ClipTooShortError it aborted whole batch
    # runs rather than skipping one sample.
    profile = ThresholdProfile(
        name="wide-area", max_flashes_per_second=3, max_area_ratio=0.50,
        general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
        red_saturation_ratio_threshold=0.80,
    )
    rng = np.random.default_rng(0)
    for _ in range(200):
        lights = params.sample_lights(profile, 480, 854, rng)
        assert len(lights) >= 1
```

Ensure the test file imports what these tests use (add only what is missing):

```python
import numpy as np

from detector.profiles import ThresholdProfile
from training import params
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest training/tests/test_params.py -k "feasible_kinds or max_achievable_area or high_area_targets" -v`
Expected: FAIL with `AttributeError: module 'training.params' has no attribute '_max_achievable_area'` (and the same for `_feasible_kinds`). `test_sample_lights_does_not_raise_at_high_area_targets` fails with `ValueError: profile 'wide-area': combined injected light area ratio ... does not exceed max_area_ratio=0.5000`.

- [ ] **Step 3: Implement `_max_achievable_area` and `_feasible_kinds`**

Insert into `training/params.py` immediately after the existing `_light_area` function:

```python
def _max_achievable_area(kind: str, frame_height: int, frame_width: int) -> int:
    """Largest area `_sample_one_light` can actually produce for `kind`.

    Every branch of `_sample_one_light` clamps its size to at most half the
    frame, so each kind has a hard ceiling well below the full frame -- and
    the ceilings differ sharply between kinds (measured on 480x854: rect
    ~0.56 of the frame, beam ~0.52, circle ~0.44). Mirrors those clamps
    exactly; `test_max_achievable_area_matches_what_sample_one_light_actually_produces`
    pins the two together so they cannot drift apart.
    """
    max_half = max(1, min(frame_height, frame_width) // 2)
    if kind == "rect":
        return mask_shapes.rect_area(max_half, max_half)
    if kind == "circle":
        return mask_shapes.circle_area(max_half)
    if kind == "beam":
        half_length = max(1, min(_BEAM_ASPECT * max_half, max(frame_height, frame_width) // 2))
        return mask_shapes.beam_area(half_length, max_half)
    raise ValueError(f"unknown light kind: {kind!r}")


def _feasible_kinds(target_area: float, frame_height: int, frame_width: int) -> tuple[str, ...]:
    """Kinds that can actually reach `target_area` on this frame.

    Picking uniformly from all kinds regardless of target was the direct
    cause of `sample_lights` raising a plain (batch-aborting) ValueError on
    a measured ~33% of draws at a 0.60 target: a kind whose ceiling sits
    below its share of the target simply cannot satisfy the combined-area
    check. Falls back to every kind when none qualifies, so an unsatisfiable
    target still surfaces as `_require_exceeds`'s explanatory ValueError
    rather than an IndexError on an empty tuple.
    """
    feasible = tuple(
        kind for kind in _LIGHT_KINDS
        if _max_achievable_area(kind, frame_height, frame_width) >= target_area
    )
    return feasible or _LIGHT_KINDS
```

- [ ] **Step 4: Wire `_feasible_kinds` into `sample_lights`**

In `training/params.py`, inside `sample_lights`, replace this block:

```python
    lights = [
        _sample_one_light(
            _LIGHT_KINDS[int(rng.integers(0, len(_LIGHT_KINDS)))],
            per_light_area, frame_height, frame_width, rng,
        )
        for _ in range(n_lights)
    ]
```

with:

```python
    kinds = _feasible_kinds(per_light_area, frame_height, frame_width)
    lights = [
        _sample_one_light(
            kinds[int(rng.integers(0, len(kinds)))],
            per_light_area, frame_height, frame_width, rng,
        )
        for _ in range(n_lights)
    ]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest training/tests/test_params.py -v`
Expected: all PASS, including the four new tests.

- [ ] **Step 6: Run the full repository test suite**

Run: `pytest -q` from the repo root.
Expected: all pass (213 pre-existing + 5 new from this task).

- [ ] **Step 7: Commit**

```bash
git add training/params.py training/tests/test_params.py
git commit -m "fix(training): only sample light kinds that can reach the target area"
```

---

## Task 2: Randomize injected mask area

**Files:**
- Modify: `training/params.py` (`sample_lights`'s `target_total_area` computation)
- Test: `training/tests/test_params.py`

**Interfaces:**
- Consumes: `_feasible_kinds` from Task 1 (already wired into `sample_lights`); existing constants `_AREA_MARGIN = 0.10`, `_MAX_LIGHTS_AREA_RATIO = 0.60`.
- Produces: no new public names. `sample_lights`'s signature is unchanged — only the distribution of the areas it returns changes.

**Background for the implementer:** `sample_lights` currently computes one fixed target area from the profile, so every training example under a given profile has essentially the same mask size. The real test footage this data is meant to represent reached 74.8% mask coverage. Sampling uniformly across `[profile.max_area_ratio + _AREA_MARGIN, _MAX_LIGHTS_AREA_RATIO]` gives small localized flashes and near-full-screen ones equal representation. When the profile's own floor already meets or exceeds the `_MAX_LIGHTS_AREA_RATIO` ceiling there is no range left to sample from, so the floor is used directly — exactly the value the current code produces.

- [ ] **Step 1: Write the failing tests**

Add to `training/tests/test_params.py`:

```python
def _area_ratio_of(lights, frame_height, frame_width):
    return sum(params._light_area(light) for light in lights) / (frame_height * frame_width)


def test_sample_lights_produces_a_range_of_areas_not_one_fixed_value():
    # The whole point of this change: before it, every draw under a profile
    # produced essentially the same total area, so the model never saw a
    # localized flash and a near-full-screen one as different problems.
    profile = ThresholdProfile(
        name="test", max_flashes_per_second=3, max_area_ratio=0.10,
        general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
        red_saturation_ratio_threshold=0.80,
    )
    rng = np.random.default_rng(0)
    frame_height, frame_width = 480, 854

    ratios = [
        _area_ratio_of(params.sample_lights(profile, frame_height, frame_width, rng), frame_height, frame_width)
        for _ in range(200)
    ]

    assert max(ratios) - min(ratios) > 0.15


def test_sample_lights_area_always_exceeds_the_profile_threshold():
    # Randomizing area must never produce a sample that isn't actually
    # hazardous -- _require_exceeds guards this, and this test pins that the
    # guard still covers every draw.
    profile = ThresholdProfile(
        name="test", max_flashes_per_second=3, max_area_ratio=0.10,
        general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
        red_saturation_ratio_threshold=0.80,
    )
    rng = np.random.default_rng(1)
    frame_height, frame_width = 480, 854

    for _ in range(200):
        lights = params.sample_lights(profile, frame_height, frame_width, rng)
        assert _area_ratio_of(lights, frame_height, frame_width) > profile.max_area_ratio


def test_sample_lights_handles_a_profile_whose_floor_meets_the_ceiling():
    # max_area_ratio 0.50 + _AREA_MARGIN 0.10 lands exactly on
    # _MAX_LIGHTS_AREA_RATIO 0.60, leaving an empty range -- must use the
    # floor rather than raising or sampling from an inverted interval.
    profile = ThresholdProfile(
        name="edge", max_flashes_per_second=3, max_area_ratio=0.50,
        general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
        red_saturation_ratio_threshold=0.80,
    )
    rng = np.random.default_rng(2)
    lights = params.sample_lights(profile, 480, 854, rng)
    assert _area_ratio_of(lights, 480, 854) > profile.max_area_ratio
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest training/tests/test_params.py -k "range_of_areas or floor_meets_the_ceiling" -v`
Expected: `test_sample_lights_produces_a_range_of_areas_not_one_fixed_value` FAILS on the spread assertion (all 200 draws land at nearly the same ratio). The other two may already pass — that is fine, they are guard tests that must keep passing after the change.

- [ ] **Step 3: Randomize the area target**

In `training/params.py`, inside `sample_lights`, replace:

```python
    target_total_area = min(
        (profile.max_area_ratio + _AREA_MARGIN) * frame_height * frame_width,
        _MAX_LIGHTS_AREA_RATIO * frame_height * frame_width,
    )
```

with:

```python
    # Sample the severity rather than fixing it at "just barely over the
    # profile's threshold": real footage ranges from a small localized
    # flash to nearly the whole frame (74.8% measured on the concert clip
    # this data is meant to represent), and a model that only ever trained
    # on threshold-hugging areas has never seen the severe case. Uniform, so
    # mild and severe get equal representation.
    min_area_ratio = profile.max_area_ratio + _AREA_MARGIN
    # A profile whose floor already meets the ceiling leaves no range to
    # sample -- use the floor, which is exactly what the pre-randomization
    # code produced.
    max_area_ratio = max(min_area_ratio, _MAX_LIGHTS_AREA_RATIO)
    target_area_ratio = float(rng.uniform(min_area_ratio, max_area_ratio))
    target_total_area = target_area_ratio * frame_height * frame_width
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest training/tests/test_params.py -v`
Expected: all PASS, including the three new tests.

- [ ] **Step 5: Run the full repository test suite**

Run: `pytest -q` from the repo root.
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add training/params.py training/tests/test_params.py
git commit -m "feat(training): sample injected mask area across a severity range"
```

---

## Task 3: Randomize brightness and color contrast

**Files:**
- Modify: `training/params.py` (both branches of `sample_synthesis_params`)
- Test: `training/tests/test_params.py`

**Interfaces:**
- Consumes: existing constants `_RATIO_MARGIN = 0.05`, `_MAX_SATURATION_TARGET = 0.99`; existing `_require_exceeds`.
- Produces: no new public names. `sample_synthesis_params`'s signature and return type (`SynthParams`) are unchanged — only the distribution of `dark_target` / `bright_target` / `red_rgb` / `baseline_rgb` changes.

**Background for the implementer:** Both pattern branches currently compute contrast from fixed formulas. The `general` branch always produces the minimum contrast that clears the profile's delta threshold. The `red` branch is worse — `r = 0.9` and `baseline_rgb = (0.3, 0.3, 0.3)` are hardcoded constants that don't even depend on the profile, so every red example ever generated is identical in color. Real footage measured on the test clip had its R channel oscillating between 0.17 and 0.50.

Two new module constants are introduced for the red branch's bounds. `_MIN_RED_TARGET = 0.5`: below this, `gb_sum = r / target_ratio - r` leaves too little total signal to render a recognisable red flash. `_MAX_RED_BASELINE = 0.5`: a baseline as bright as the red pulse erases the contrast the pattern depends on.

- [ ] **Step 1: Write the failing tests**

Add to `training/tests/test_params.py`:

```python
_CONTRAST_PROFILE = ThresholdProfile(
    name="contrast", max_flashes_per_second=3, max_area_ratio=0.10,
    general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
    red_saturation_ratio_threshold=0.80,
)


def _sample_many(pattern, n=200, seed=0):
    rng = np.random.default_rng(seed)
    return [
        params.sample_synthesis_params(
            _CONTRAST_PROFILE, pattern, clip_frame_count=200,
            frame_height=480, frame_width=854, fps=24.0, rng=rng,
        )
        for _ in range(n)
    ]


def test_general_pattern_contrast_is_sampled_not_fixed():
    # Before this change every general-pattern example under a profile had
    # byte-identical dark/bright targets -- the minimum contrast that just
    # clears the threshold, never anything more severe.
    samples = _sample_many("general")
    assert len({s.dark_target for s in samples}) > 50
    assert len({s.bright_target for s in samples}) > 50


def test_general_pattern_contrast_always_clears_the_delta_threshold():
    # Randomizing must not let a sample fall below the profile's own
    # threshold -- it would be a "hazard" the Detector never flags.
    for s in _sample_many("general", seed=1):
        assert 0.0 <= s.dark_target <= _CONTRAST_PROFILE.general_flash_dark_threshold * 0.5
        assert s.bright_target <= 1.0
        assert s.bright_target - s.dark_target > _CONTRAST_PROFILE.general_flash_delta_threshold


def test_red_pattern_color_is_sampled_not_hardcoded():
    # r was a hardcoded 0.9 and baseline a hardcoded (0.3, 0.3, 0.3),
    # independent of profile -- every red example in the entire dataset was
    # the same color against the same gray.
    samples = _sample_many("red", seed=2)
    assert len({s.red_rgb[0] for s in samples}) > 50
    assert len({s.baseline_rgb[0] for s in samples}) > 50


def test_red_pattern_stays_within_its_bounds_and_clears_the_threshold():
    for s in _sample_many("red", seed=3):
        r, g, b = s.red_rgb
        assert params._MIN_RED_TARGET <= r <= _MAX_SATURATION_TARGET
        base = s.baseline_rgb[0]
        assert 0.0 <= base <= params._MAX_RED_BASELINE
        assert s.baseline_rgb == (base, base, base)
        # The saturation ratio the injected red actually achieves must
        # exceed the profile's threshold, or the sample isn't hazardous.
        assert r / (r + g + b) > _CONTRAST_PROFILE.red_saturation_ratio_threshold
```

Add this import alongside the existing ones in the test file if not already present:

```python
from training.params import _MAX_SATURATION_TARGET
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest training/tests/test_params.py -k "contrast_is_sampled or color_is_sampled or within_its_bounds" -v`
Expected: `test_general_pattern_contrast_is_sampled_not_fixed` and `test_red_pattern_color_is_sampled_not_hardcoded` FAIL (only 1 distinct value, not >50). `test_red_pattern_stays_within_its_bounds_and_clears_the_threshold` FAILS with `AttributeError: module 'training.params' has no attribute '_MIN_RED_TARGET'`.

- [ ] **Step 3: Add the red-branch bound constants**

In `training/params.py`, add next to the existing `_MAX_SATURATION_TARGET` definition:

```python
# Below this the red channel carries too little signal for `gb_sum` to leave
# a recognisable red flash once the saturation ratio is applied.
_MIN_RED_TARGET = 0.5
# A baseline as bright as the red pulse itself erases the very contrast the
# red pattern depends on.
_MAX_RED_BASELINE = 0.5
```

- [ ] **Step 4: Randomize the `general` branch**

In `training/params.py`, inside `sample_synthesis_params`, replace:

```python
    if pattern == "general":
        dark_target = profile.general_flash_dark_threshold * 0.5
        bright_target = min(1.0, dark_target + profile.general_flash_delta_threshold + 0.1)
```

with:

```python
    if pattern == "general":
        # Sample the contrast instead of always emitting the mildest
        # threshold-clearing pair. The previous fixed values sit at the ends
        # of these ranges (darkest-allowed dark, mildest-allowed bright), so
        # this only widens what gets generated -- it never drops what used
        # to work.
        dark_target = float(rng.uniform(0.0, profile.general_flash_dark_threshold * 0.5))
        min_bright = dark_target + profile.general_flash_delta_threshold + 0.1
        bright_target = float(rng.uniform(min_bright, 1.0)) if min_bright < 1.0 else 1.0
```

- [ ] **Step 5: Randomize the `red` branch**

In `training/params.py`, inside `sample_synthesis_params`, replace:

```python
    target_ratio = min(_MAX_SATURATION_TARGET, profile.red_saturation_ratio_threshold + _RATIO_MARGIN)
```

with:

```python
    # Sample the saturation ratio across everything above the profile's
    # threshold rather than sitting one fixed margin above it.
    min_ratio = min(_MAX_SATURATION_TARGET, profile.red_saturation_ratio_threshold + _RATIO_MARGIN)
    target_ratio = float(rng.uniform(min_ratio, _MAX_SATURATION_TARGET)) if min_ratio < _MAX_SATURATION_TARGET else min_ratio
```

and replace:

```python
    r = 0.9
    gb_sum = r / target_ratio - r
    red_rgb = (r, gb_sum / 2, gb_sum / 2)
    baseline_rgb = (0.3, 0.3, 0.3)
```

with:

```python
    # r and the baseline were hardcoded constants -- not even
    # profile-derived -- so every red sample ever generated was the same
    # color against the same gray. Real footage varies both (R measured
    # oscillating 0.17-0.50 on the concert clip).
    r = float(rng.uniform(_MIN_RED_TARGET, _MAX_SATURATION_TARGET))
    gb_sum = r / target_ratio - r
    red_rgb = (r, gb_sum / 2, gb_sum / 2)
    baseline_value = float(rng.uniform(0.0, _MAX_RED_BASELINE))
    baseline_rgb = (baseline_value, baseline_value, baseline_value)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest training/tests/test_params.py -v`
Expected: all PASS, including the four new tests.

- [ ] **Step 7: Run the full repository test suite**

Run: `pytest -q` from the repo root.
Expected: all pass. If `training/tests/test_synth.py` fails because a test asserted an exact `dark_target`/`bright_target`/`red_rgb` value that is now sampled, update that assertion to a range check rather than reverting the randomization — report the change in your task report.

- [ ] **Step 8: Commit**

```bash
git add training/params.py training/tests/test_params.py
git commit -m "feat(training): sample brightness and color contrast per example"
```

---

## Task 4: Guard the batch loop against per-sample ValueError

**Files:**
- Modify: `training/cli.py:170-180` (the `try`/`except ClipTooShortError` block in the synthesis loop)
- Test: `training/tests/test_cli.py`

**Interfaces:**
- Consumes: existing `_record_failure(summary, reason, clip_id, profile_name=None, pattern=None, index=None, error=None) -> None`; existing `ClipTooShortError` (a `ValueError` subclass).
- Produces: no new names. Adds a `"synthesis_error"` value to the `reason` field already present in `summary["failed_details"]`.

**Background for the implementer:** `ClipTooShortError` subclasses `ValueError`, and the loop catches it specifically so one unusable clip is skipped rather than killing the run. Any *other* `ValueError` — for example `_require_exceeds` rejecting an unsatisfiable profile — propagates and aborts the whole batch, potentially partway through a multi-hour run. Task 1 removes the known cause, but the fragile shape remains: a single bad sample should never destroy a long run's remaining work.

Order matters in `except` clauses: `ClipTooShortError` must be caught before the broader `ValueError`, or the subclass branch becomes unreachable.

- [ ] **Step 1: Write the failing test**

Add to `training/tests/test_cli.py`:

```python
def test_run_batch_records_a_synthesis_valueerror_instead_of_aborting(tmp_path, monkeypatch):
    # A single unsatisfiable sample must not destroy the rest of a
    # multi-hour batch run. ClipTooShortError was already handled, but any
    # other ValueError (e.g. _require_exceeds rejecting a profile) escaped
    # and aborted everything.
    import training.cli as cli_module

    calls = {"n": 0}

    def _explode_once(clean_frames, fps, profile, pattern, rng):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("synthetic failure for test")
        return None

    monkeypatch.setattr(cli_module, "synthesize_sample_realistic", _explode_once)
    monkeypatch.setattr(cli_module, "synthesize_sample", _explode_once)

    clips_dir, profiles_dir = _standard_inputs(tmp_path)

    summary = run_batch(clips_dir, profiles_dir, tmp_path / "out", samples_per_combo=1, seed=0)

    assert calls["n"] > 1  # kept going past the raising sample
    assert any(d["reason"] == "synthesis_error" for d in summary["failed_details"])
```

This reuses `_standard_inputs(tmp_path)` and the `run_batch(clips_dir, profiles_dir, out_dir, samples_per_combo=..., seed=...)` positional signature already used throughout that file; `run_batch` is already imported there.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest training/tests/test_cli.py -k synthesis_valueerror -v`
Expected: FAIL with `ValueError: synthetic failure for test` propagating out of `run_batch`.

- [ ] **Step 3: Add the guard**

In `training/cli.py`, replace:

```python
                    except ClipTooShortError as exc:
                        _record_failure(
                            summary, "clip_too_short", clip_id,
                            profile.name, pattern, index, str(exc),
                        )
                        continue
```

with:

```python
                    except ClipTooShortError as exc:
                        _record_failure(
                            summary, "clip_too_short", clip_id,
                            profile.name, pattern, index, str(exc),
                        )
                        continue
                    except ValueError as exc:
                        # ClipTooShortError (a ValueError subclass) is caught
                        # above; this catches everything else, so one
                        # unsatisfiable sample is recorded and skipped rather
                        # than aborting a long run partway through with all
                        # remaining work lost.
                        _record_failure(
                            summary, "synthesis_error", clip_id,
                            profile.name, pattern, index, str(exc),
                        )
                        continue
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest training/tests/test_cli.py -v`
Expected: all PASS, including the new test.

- [ ] **Step 5: Run the full repository test suite**

Run: `pytest -q` from the repo root.
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add training/cli.py training/tests/test_cli.py
git commit -m "fix(training): skip a failing sample instead of aborting the batch"
```

---

## Task 5: Regenerate the dataset and measure the resulting distributions

**Files:**
- Modify: `docs/superpowers/specs/2026-08-04-synthesis-parameter-diversification-design.md` (append a measured-results section)
- No code changes.

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: measured before/after distribution numbers, recorded in the design doc.

**Background for the implementer:** The unit tests prove each sampler's range in isolation. This task proves the change actually lands in generated data, and that widening the parameters did not collapse the Detector re-validation acceptance rate (`synthesize_sample*` retries up to `max_retries` and returns `None` when the injected pattern isn't detected — a sample that fails all retries is dropped). This is the same real-regeneration verification the previous DatasetSynth change used.

- [ ] **Step 1: Delete stale prior caches**

Prior caches under `data/synthetic` were computed from the old parameters and would be silently reused by the next training run.

Run:
```bash
find data/synthetic -name "prior_cache.npz" -delete
```

- [ ] **Step 2: Regenerate the dataset**

Run from the repo root (this takes a while — it re-synthesizes every sample):

```bash
python -m training.cli --clips-dir data/clips --output data/synthetic --overwrite --summary-output regen_summary.json
```

- [ ] **Step 3: Check the acceptance rate did not collapse**

Read `regen_summary.json` and compare `accepted` against `failed`. Note the counts and the `failed_details` reason breakdown.

Expected: acceptance stays comparable to before this change. Widening parameters mostly makes patterns *more* severe and therefore easier for the Detector to flag, so acceptance should hold or improve. A large drop, or any `"synthesis_error"` entries, means something in Tasks 1-3 is producing invalid parameters — report it rather than accepting the result.

- [ ] **Step 4: Measure the achieved distributions**

Run:

```bash
python -c "
import json
from pathlib import Path
import numpy as np

areas = []
for sample_dir in sorted(Path('data/synthetic').iterdir()):
    meta_path = sample_dir / 'meta.json'
    if not meta_path.exists():
        continue
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    lights = meta.get('window', {}).get('lights') or []
    if lights:
        areas.append(len(lights))
print('samples with lights:', len(areas))
print('lights-per-sample: min', min(areas), 'max', max(areas))
"
```

Then measure the actual rendered mask-area spread by sampling the written frames:

```bash
python -c "
import json
from pathlib import Path
import numpy as np
from detector.profiles import load_profile
from prior.compute import compute_prior
import cv2

ratios = []
dirs = [d for d in sorted(Path('data/synthetic').iterdir()) if (d / 'meta.json').exists()]
for sample_dir in dirs[::10]:
    meta = json.loads((sample_dir / 'meta.json').read_text(encoding='utf-8'))
    degraded = sorted((sample_dir / 'degraded').glob('*.png'))
    if not degraded:
        continue
    frames = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for p in degraded]
    profile = load_profile(Path('configs/profiles') / (meta['profile'] + '.json'))
    priors = compute_prior(frames, fps=meta['fps'], profile=profile)
    peak = max(p.mask.mean() for p in priors)
    ratios.append(peak)
r = np.array(ratios)
print('samples measured:', len(r))
print('peak mask area ratio: min %.3f  median %.3f  max %.3f' % (r.min(), np.median(r), r.max()))
"
```

**Note for the implementer:** the profile-file lookup above assumes `meta.json`'s profile name matches a filename in `configs/profiles/`. If the field name or the mapping differs, adjust the snippet to read the profile the way `training/mitigator_dataset.py` does. Report the numbers you get either way.

Expected: a visibly wider spread of peak mask area than the pre-change data, which clustered near each profile's `max_area_ratio`.

- [ ] **Step 5: Record the measurements in the design doc**

Append a `## 8. 실측 결과` section to `docs/superpowers/specs/2026-08-04-synthesis-parameter-diversification-design.md` containing:
- the acceptance/failure counts from Step 3,
- the mask-area distribution numbers from Step 4,
- one sentence stating whether §6's "재합성 후 분포 확대 확인" criterion is met.

Write actual measured numbers. If a measurement could not be taken, say so explicitly and why — do not write a placeholder.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-08-04-synthesis-parameter-diversification-design.md
git commit -m "docs(spec): record measured distributions after diversification"
```

Do **not** commit `data/synthetic` (it is gitignored) or `regen_summary.json`. Delete `regen_summary.json` after recording its numbers.

---

## Self-Review Notes (for the plan author, not a task)

**Spec coverage check:**
- §3.1/§3.2 area randomization → Task 2
- §3.3 shape-kind feasibility fix (2-step: root cause + batch guard) → Task 1 (root cause), Task 4 (guard)
- §4.1 general contrast → Task 3 Step 4
- §4.2 red contrast → Task 3 Steps 3, 5
- §4.3 `_require_exceeds` retained → verified by Task 2's `test_sample_lights_area_always_exceeds_the_profile_threshold` and Task 3's threshold-clearing tests; no task removes any `_require_exceeds` call
- §5 error handling → Task 4
- §6 unit tests (range, variety, threshold) → Tasks 1-3; regression tests → Task 1 Step 1, Task 4; integration/regeneration → Task 5
- §7 success criteria → Task 5 Step 5. The project-level goal (`run_detection` finding zero risk segments on corrected output) is explicitly **not** a criterion for this plan — §7 states this plan alone likely won't reach it, and retraining/re-evaluation happens after this work.

**Verified against real code, not guessed:** the exact bodies being replaced in Tasks 1-4 were read from the current `training/params.py` and `training/cli.py`; `test_cli.py`'s helper names and `run_batch`'s positional signature in Task 4 were read from the real test file; the per-kind achievable-area figures quoted in Task 1 were computed by running `mask_shapes`' area functions against `_sample_one_light`'s clamps.

**Known limitation carried forward, not hidden:** Task 5's measurement snippets assume `meta.json` has a `profile` field naming a file in `configs/profiles/` and a `window.lights` list. This was not verified against a real `meta.json`. Both snippets are flagged inline with instructions to adapt to the real field names and report the numbers either way, rather than silently skipping the measurement.
