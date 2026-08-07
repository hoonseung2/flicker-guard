# Self-Supervised Loss Implementation Plan (Plan A of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the self-supervised loss — fidelity to the input, temporal consistency under global-shift compensation, and a differentiable relaxation of the Detector's own risk criterion — without touching the training pipeline.

**Architecture:** Three additions. A confidence guard inside `detector.motion.estimate_global_shift` so a garbage shift never reaches anything. A new torch-only module `training/soft_risk.py` that mirrors `detector.flash`'s two masks through sigmoids at the same thresholds. New functions in `training/mitigator_losses.py` that assemble the three terms. Nothing in `prior/`, `mitigator/arch.py`, or the dataset changes here — that is Plan B.

**Tech Stack:** Python 3.12, PyTorch, NumPy, OpenCV (`cv2`), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-07-self-supervised-mitigation-design.md` (sections 3 and 8)

## Global Constraints

- Frames are RGB `float32` in `[0, 1]`, shape `(H, W, 3)` for NumPy and `(B, 3, H, W)` for torch.
- The relaxation must use the **same thresholds and the same combining structure** as `detector.flash`: `transition_mask` is `(darker < general_flash_dark_threshold) & (delta > general_flash_delta_threshold)`, `red_flash_mask` is `is_saturated_red(prev) != is_saturated_red(curr)` with `R/(R+G+B) >= red_saturation_ratio_threshold`, and `detector.pipeline` combines them with `|`.
- `detector/` stays NumPy-only. Torch lives in `training/`. Do not import torch from `detector/`.
- Motion compensation is global shift only, from `detector.motion` — never dense optical flow. The loss must make the same motion error the Detector makes, or it optimises something the Detector never measures.
- Run tests from the repo root: `python -m pytest`. `pytest.ini` sets `pythonpath = .` and defaults to `-m "not slow"`.
- Existing tests must keep passing. The suite is 284 passed / 1 deselected at the base commit.
- Default hyperparameters from the spec: `λ_t = 1.0`, `λ_r = 1.0`, `τ = 0.02`, `τ_a = 0.05`.

---

### Task 1: Confidence guard on global shift estimation

`cv2.phaseCorrelate` returns `(shift, response)`. `detector/motion.py` discards the response, so a shift with no structure behind it is applied as if it were sound. Measured on 11 real clips at 480px: peak shift **1392.7 px on a 480-pixel frame**, response −0.000.

**Files:**
- Modify: `detector/motion.py`
- Test: `detector/tests/test_motion.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `estimate_global_shift(prev_gray, curr_gray, min_response=0.30) -> tuple[float, float]`. Returns `(0.0, 0.0)` when the response falls below `min_response`. Passing `min_response=0.0` restores the previous behaviour exactly.

- [ ] **Step 1: Write the failing tests**

Append to `detector/tests/test_motion.py`:

```python
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


def test_guard_rejects_a_shift_on_a_repeating_pattern():
    # A periodic pattern fits equally well one period over, so the
    # correlation peak splits and the reported shift is arbitrary. This is
    # the failure mode found on real footage (clip 57523: 299 of 299 frame
    # pairs unusable, despite the mildest flicker in the set and no cuts).
    x = np.arange(128, dtype=np.float32)
    stripes = np.tile(np.sin(x * np.pi / 4)[None, :], (128, 1)).astype(np.float32)
    shifted = np.roll(stripes, shift=17, axis=1)

    assert estimate_global_shift(stripes, shifted) == (0.0, 0.0)


def test_min_response_zero_restores_the_unguarded_behaviour():
    # Callers that want the raw estimate (e.g. a diagnostic that plots the
    # distribution) must be able to opt out without reimplementing it.
    rng = np.random.default_rng(2)
    a = rng.random((128, 128)).astype(np.float32)
    b = rng.random((128, 128)).astype(np.float32)

    assert estimate_global_shift(a, b, min_response=0.0) != (0.0, 0.0)
```

If `detector/tests/test_motion.py` does not already import `numpy as np` and `estimate_global_shift`, add those imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest detector/tests/test_motion.py -v`
Expected: FAIL — the noise and stripe cases return a non-zero shift, and `min_response` is an unexpected keyword argument.

- [ ] **Step 3: Implement**

Replace `estimate_global_shift` in `detector/motion.py` with:

```python
# Below this correlation-peak height, phase correlation has not locked onto
# anything and the shift it reports is noise. Measured across 11 real clips
# at 480px: sound estimates score 0.35-0.84, garbage 0.03-0.23, and 0.30
# sits in the gap. The largest garbage shift observed was 1392.7 px on a
# 480-pixel frame, at response -0.000.
#
# The usual cause is not flicker -- injected flicker costs at most 0.99 px
# of error even at 80% frame area (spec section 8) -- but repetitive
# structure: a periodic pattern fits equally well one period over, so the
# peak splits.
_MIN_PHASE_CORRELATION_RESPONSE = 0.30


def estimate_global_shift(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    min_response: float = _MIN_PHASE_CORRELATION_RESPONSE,
) -> tuple[float, float]:
    """Estimate global shift (pan) between two grayscale frames using phase correlation.

    Returns (0.0, 0.0) when the correlation peak is too weak to trust.
    Refusing to compensate is the pre-motion-stage behaviour and never
    compares unrelated pixels; applying an unfounded shift does.

    Args:
        prev_gray: Previous frame as (H, W) float32 grayscale array.
        curr_gray: Current frame as (H, W) float32 grayscale array.
        min_response: Reject the estimate below this peak height. Pass 0.0
            to take the raw estimate regardless.

    Returns:
        Tuple of (dx, dy) representing pixel shift in x and y directions.
    """
    (dx, dy), response = cv2.phaseCorrelate(
        prev_gray.astype(np.float32), curr_gray.astype(np.float32)
    )
    if response < min_response:
        return 0.0, 0.0
    return dx, dy
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS, 288 tests (284 + 4 new).

If a `detector/` or `prior/` test now fails, do **not** relax the threshold to make it pass. A test that depended on an ungrounded shift was asserting the old bug. Report it and stop — the controller decides.

- [ ] **Step 5: Commit**

```bash
git add detector/motion.py detector/tests/test_motion.py
git commit -m "fix(detector): reject phase-correlation shifts with no structure behind them"
```

---

### Task 2: Differentiable relaxation of the Detector's risk criterion

**Files:**
- Create: `training/soft_risk.py`
- Test: `training/tests/test_soft_risk.py`

**Interfaces:**
- Consumes: `training.mitigator_losses.relative_luminance_torch(frames: Tensor) -> Tensor` — takes `(B, 3, H, W)` sRGB, returns `(B, 1, H, W)` relative luminance; already exists (added 2026-08-06).
- Produces:
  - `soft_flagged_area(prev_rgb, curr_rgb, profile, tau=0.02, valid_mask=None) -> Tensor` — per-batch-item scalar, shape `(B,)`, in `[0, 1]`.
  - `risk_penalty(area, max_area_ratio, tau_area=0.05) -> Tensor` — same shape as `area`.

- [ ] **Step 1: Write the failing tests**

Create `training/tests/test_soft_risk.py`:

```python
import numpy as np
import pytest
import torch

from detector.flash import red_flash_mask, transition_mask
from detector.luminance import relative_luminance
from detector.profiles import ThresholdProfile
from training.soft_risk import risk_penalty, soft_flagged_area

PROFILE = ThresholdProfile(
    name="test", max_flashes_per_second=3, max_area_ratio=0.25,
    general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
    red_saturation_ratio_threshold=0.80,
)


def _to_torch(frame_hwc):
    return torch.from_numpy(frame_hwc.transpose(2, 0, 1)).float()[None]


def _hard_area(prev_hwc, curr_hwc):
    """The Detector's own flagged-area ratio, computed with numpy."""
    mask = transition_mask(
        relative_luminance(prev_hwc), relative_luminance(curr_hwc),
        dark_threshold=PROFILE.general_flash_dark_threshold,
        delta_threshold=PROFILE.general_flash_delta_threshold,
    ) | red_flash_mask(prev_hwc, curr_hwc,
                       saturation_ratio_threshold=PROFILE.red_saturation_ratio_threshold)
    return float(mask.mean())


def test_identical_frames_have_almost_no_flagged_area():
    frame = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32) * 0.5
    area = soft_flagged_area(_to_torch(frame), _to_torch(frame), PROFILE)
    assert area.shape == (1,)
    assert area.item() < 0.01


def test_soft_area_tracks_the_detector_hard_area():
    # The whole point of the relaxation: it must measure the same quantity
    # the Detector thresholds on. If these diverge, the loss optimises
    # something the pass/fail rule does not care about -- the exact failure
    # this design exists to remove.
    rng = np.random.default_rng(1)
    base = rng.random((64, 64, 3)).astype(np.float32) * 0.2
    for fraction in (0.10, 0.25, 0.50, 0.80):
        bright = base.copy()
        rows = int(64 * fraction)
        bright[:rows] = np.clip(bright[:rows] + 0.7, 0.0, 1.0)

        hard = _hard_area(base, bright)
        soft = soft_flagged_area(_to_torch(base), _to_torch(bright), PROFILE).item()
        assert abs(soft - hard) < 0.05, f"fraction {fraction}: soft {soft:.3f} vs hard {hard:.3f}"


def test_red_flash_is_detected_without_a_luminance_change():
    # red_flash_mask fires on a change in "is this pixel saturated red",
    # independent of brightness. A relaxation that only softened the
    # luminance test would score this pair as safe.
    rng = np.random.default_rng(2)
    grey = rng.random((32, 32, 3)).astype(np.float32) * 0.1 + 0.3
    red = grey.copy()
    red[:, :, 0] = 0.9
    red[:, :, 1] = 0.02
    red[:, :, 2] = 0.02

    assert soft_flagged_area(_to_torch(grey), _to_torch(red), PROFILE).item() > 0.5


def test_gradients_flow_to_both_frames():
    rng = np.random.default_rng(3)
    a = _to_torch(rng.random((16, 16, 3)).astype(np.float32) * 0.3).requires_grad_(True)
    b = _to_torch(rng.random((16, 16, 3)).astype(np.float32) * 0.9).requires_grad_(True)

    soft_flagged_area(a, b, PROFILE).sum().backward()
    assert torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0
    assert torch.isfinite(b.grad).all() and b.grad.abs().sum() > 0


def test_valid_mask_excludes_pixels_from_the_average():
    # The temporal term warps one frame, and BORDER_REPLICATE fills the
    # edge with copies that are not real correspondences. Those pixels must
    # not count toward the area, or a wide warp dilutes it toward zero and
    # the model is under-charged for exceeding the limit.
    rng = np.random.default_rng(4)
    base = rng.random((32, 32, 3)).astype(np.float32) * 0.2
    bright = np.clip(base + 0.7, 0.0, 1.0)

    valid = torch.ones(1, 1, 32, 32)
    valid[:, :, :, :16] = 0.0   # discard the left half

    full = soft_flagged_area(_to_torch(base), _to_torch(bright), PROFILE).item()
    half = soft_flagged_area(_to_torch(base), _to_torch(bright), PROFILE, valid_mask=valid).item()
    assert full == pytest.approx(half, abs=0.05)   # uniformly flagged -> same ratio


def test_penalty_is_negligible_below_the_limit_and_grows_above():
    below = risk_penalty(torch.tensor([0.10]), max_area_ratio=0.25).item()
    at = risk_penalty(torch.tensor([0.25]), max_area_ratio=0.25).item()
    above = risk_penalty(torch.tensor([0.60]), max_area_ratio=0.25).item()

    assert below < 0.01
    assert below < at < above
    assert above > 0.3


def test_penalty_keeps_a_gradient_below_the_limit():
    # softplus rather than relu: a model sitting just under the limit still
    # feels a small pull toward more headroom, instead of a flat zero that
    # lets it drift back over.
    area = torch.tensor([0.10], requires_grad=True)
    risk_penalty(area, max_area_ratio=0.25).backward()
    assert area.grad.abs().item() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest training/tests/test_soft_risk.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.soft_risk'`

- [ ] **Step 3: Implement**

Create `training/soft_risk.py`:

```python
"""A differentiable stand-in for the Detector's risk criterion.

`detector.segments._is_risky` fires on hard thresholds, which have zero
gradient, so a loss cannot be built from it directly. This module relaxes
each threshold through a sigmoid while keeping the SAME threshold values and
the SAME combining structure as `detector.flash` -- the general-flash test
ANDed from a delta and a darkness condition, the red-flash test as a change
in "is this pixel saturated red", the two ORed together.

That correspondence is the point. Every failure this project has measured
traces to the loss and the pass/fail rule measuring different things: 35.66
dB on synthetic data against risk segments still standing on real footage,
amplitude down 60% with the flicker rate down 12%, the model reaching a
median 14.6% of its own conditioning target. A relaxation that drifts from
the Detector's thresholds would reproduce exactly that.

Torch lives here rather than in `detector/` so the classical detection path
keeps its NumPy-only dependency footprint.

The frequency half of `_is_risky` is deliberately not relaxed.
`detector.scoring.WindowedFlashCounter` sets `had_flag = area_ratio > 0.0`,
so a single pixel over threshold flags the whole frame and the frequency
count saturates at the frame rate on real footage -- measured across
wonbon's full length and Cera Khin's first nine seconds. ANDing a term that
is always true would multiply the penalty by one.
"""
import torch
import torch.nn.functional as F

from detector.profiles import ThresholdProfile
from training.mitigator_losses import relative_luminance_torch

_EPS = 1e-6


def _soft_saturated_red(frames_rgb: torch.Tensor, threshold: float, tau: float) -> torch.Tensor:
    """Relaxed `detector.flash._is_saturated_red`: R / (R + G + B) >= threshold."""
    total = frames_rgb.sum(dim=1, keepdim=True).clamp(min=_EPS)
    ratio = frames_rgb[:, 0:1] / total
    return torch.sigmoid((ratio - threshold) / tau)


def soft_flagged_area(
    prev_rgb: torch.Tensor,
    curr_rgb: torch.Tensor,
    profile: ThresholdProfile,
    tau: float = 0.02,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Relaxed flagged-area ratio between two (B, 3, H, W) sRGB frames.

    Returns one scalar per batch item. `valid_mask` is an optional
    (B, 1, H, W) tensor of weights in [0, 1]; pixels weighted 0 are left out
    of the average, which is how the caller discards the replicated border a
    warp leaves behind.
    """
    prev_luminance = relative_luminance_torch(prev_rgb)
    curr_luminance = relative_luminance_torch(curr_rgb)

    delta = (curr_luminance - prev_luminance).abs()
    darker = torch.minimum(prev_luminance, curr_luminance)
    general = torch.sigmoid((delta - profile.general_flash_delta_threshold) / tau) * torch.sigmoid(
        (profile.general_flash_dark_threshold - darker) / tau
    )

    red = (
        _soft_saturated_red(prev_rgb, profile.red_saturation_ratio_threshold, tau)
        - _soft_saturated_red(curr_rgb, profile.red_saturation_ratio_threshold, tau)
    ).abs()

    # `maximum` is the relaxation of the pipeline's `|`: a pixel counts once
    # whether one test fires or both.
    flagged = torch.maximum(general, red)

    if valid_mask is None:
        return flagged.mean(dim=(1, 2, 3))
    weight = valid_mask.sum(dim=(1, 2, 3)).clamp(min=_EPS)
    return (flagged * valid_mask).sum(dim=(1, 2, 3)) / weight


def risk_penalty(
    area: torch.Tensor, max_area_ratio: float, tau_area: float = 0.05
) -> torch.Tensor:
    """Charge only for the part of `area` that exceeds the profile's limit.

    softplus rather than relu so a model sitting just under the limit still
    feels a small pull toward headroom instead of a flat zero it can drift
    back across.
    """
    return F.softplus((area - max_area_ratio) / tau_area) * tau_area
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest training/tests/test_soft_risk.py -v`
Expected: PASS, 7 tests.

If `test_soft_area_tracks_the_detector_hard_area` fails, the tolerance is not the thing to change — the relaxation is wrong somewhere. Print `soft` and `hard` per fraction: a constant offset points at the red term firing when it should not, and a widening gap at `tau` being too large for the threshold it softens.

- [ ] **Step 5: Commit**

```bash
git add training/soft_risk.py training/tests/test_soft_risk.py
git commit -m "feat(training): differentiable relaxation of the Detector's risk criterion"
```

---

### Task 3: Warping a batch by a global shift

The temporal term compares frame *i* to frame *i+1* after cancelling camera motion, so it needs `compensate_shift`'s behaviour on a torch tensor, differentiably. It also needs to know which pixels the warp invented.

**Files:**
- Modify: `training/mitigator_losses.py`
- Test: `training/tests/test_mitigator_losses.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `warp_by_shift(frames, dx, dy) -> tuple[Tensor, Tensor]` — takes `(B, C, H, W)` and per-item shifts `(B,)`, returns the warped tensor and a `(B, 1, H, W)` validity mask that is 0 where the sample fell outside the source.

- [ ] **Step 1: Write the failing tests**

Append to `training/tests/test_mitigator_losses.py`:

```python
def test_warp_by_shift_matches_compensate_shift_for_integer_shifts():
    # The loss must move pixels the same way the Detector does, or it scores
    # a different alignment than the one the pass/fail rule was computed on.
    from detector.motion import compensate_shift
    from training.mitigator_losses import warp_by_shift

    frame = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    expected = compensate_shift(frame, 3.0, -2.0)

    warped, _valid = warp_by_shift(
        torch.from_numpy(frame)[None, None], torch.tensor([3.0]), torch.tensor([-2.0])
    )
    # Compare the interior only; the two disagree at the border by
    # construction (cv2 replicates, grid_sample zero-fills and the validity
    # mask marks it), and that difference is what `valid` exists to express.
    assert np.allclose(warped[0, 0, 4:-4, 4:-4].numpy(), expected[4:-4, 4:-4], atol=1e-4)


def test_warp_by_shift_zero_is_the_identity():
    from training.mitigator_losses import warp_by_shift

    frame = torch.rand(2, 3, 16, 16)
    warped, valid = warp_by_shift(frame, torch.zeros(2), torch.zeros(2))
    assert torch.allclose(warped, frame, atol=1e-5)
    assert valid.min() > 0.99


def test_warp_by_shift_marks_the_invented_border_invalid():
    from training.mitigator_losses import warp_by_shift

    frame = torch.rand(1, 1, 32, 32)
    _warped, valid = warp_by_shift(frame, torch.tensor([6.0]), torch.tensor([0.0]))
    assert valid[0, 0, :, 0].max() < 0.5     # left column came from outside
    assert valid[0, 0, :, -1].min() > 0.5    # right column is real


def test_warp_by_shift_is_differentiable():
    from training.mitigator_losses import warp_by_shift

    frame = torch.rand(1, 3, 16, 16, requires_grad=True)
    warped, _valid = warp_by_shift(frame, torch.tensor([2.5]), torch.tensor([-1.5]))
    warped.sum().backward()
    assert torch.isfinite(frame.grad).all()
    assert frame.grad.abs().sum() > 0


def test_warp_by_shift_handles_per_item_shifts_in_a_batch():
    from training.mitigator_losses import warp_by_shift

    frame = torch.rand(1, 1, 16, 16).repeat(2, 1, 1, 1)
    warped, _valid = warp_by_shift(frame, torch.tensor([0.0, 4.0]), torch.zeros(2))
    assert torch.allclose(warped[0], frame[0], atol=1e-5)
    assert not torch.allclose(warped[1], frame[1], atol=1e-2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest training/tests/test_mitigator_losses.py -v`
Expected: FAIL — `ImportError: cannot import name 'warp_by_shift'`

- [ ] **Step 3: Implement**

Append to `training/mitigator_losses.py`:

```python
def warp_by_shift(
    frames: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate `frames` by a per-item global shift, differentiably.

    This is the torch counterpart of `detector.motion.compensate_shift`, and
    it must stay one: the temporal loss compares two frames after cancelling
    camera motion, and if it cancels a different motion than the Detector
    did, it scores an alignment the pass/fail rule never saw.

    `compensate_shift` fills the vacated edge with BORDER_REPLICATE, which
    invents correspondences that do not exist. Here the edge is zero-filled
    and reported through the returned validity mask instead, so a caller can
    exclude it rather than average fabricated pixels into a score.

    Args:
        frames: (B, C, H, W).
        dx, dy: (B,) shifts in pixels, same sign convention as
            `compensate_shift` (positive dx moves content right).

    Returns:
        (warped, valid) where `valid` is (B, 1, H, W), 1 where the sample
        came from inside the source and 0 where it did not.
    """
    batch, _channels, height, width = frames.shape
    device, dtype = frames.device, frames.dtype

    # affine_grid works in normalised [-1, 1] coordinates, and its theta maps
    # OUTPUT coordinates to INPUT ones -- so a shift of +dx pixels of content
    # to the right means sampling dx pixels to the LEFT, hence the negation.
    theta = torch.zeros(batch, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = -2.0 * dx.to(device=device, dtype=dtype) / max(width - 1, 1)
    theta[:, 1, 2] = -2.0 * dy.to(device=device, dtype=dtype) / max(height - 1, 1)

    grid = F.affine_grid(theta, (batch, 1, height, width), align_corners=True)
    warped = F.grid_sample(frames, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    ones = torch.ones(batch, 1, height, width, device=device, dtype=dtype)
    valid = F.grid_sample(ones, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return warped, valid
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest training/tests/test_mitigator_losses.py -v`
Expected: PASS — the file's existing tests plus 5 new ones.

If `test_warp_by_shift_matches_compensate_shift_for_integer_shifts` fails, the sign convention is the first thing to check: flip the negation on `theta[:, 0, 2]` / `theta[:, 1, 2]` and see whether it matches. Do not widen the interior crop to make it pass — a sign error would silently double every motion error in the loss.

- [ ] **Step 5: Commit**

```bash
git add training/mitigator_losses.py training/tests/test_mitigator_losses.py
git commit -m "feat(training): differentiable global-shift warp with a validity mask"
```

---

### Task 4: The self-supervised loss

**Files:**
- Modify: `training/mitigator_losses.py`
- Test: `training/tests/test_mitigator_losses.py`

**Interfaces:**
- Consumes: `warp_by_shift` (Task 3), `training.soft_risk.soft_flagged_area` and `risk_penalty` (Task 2), `relative_luminance_torch` (existing).
- Produces:
  ```python
  def self_supervised_loss(
      out_a, out_b, in_a, in_b, dx, dy, profile,
      lambda_temporal=1.0, lambda_risk=1.0, tau=0.02, tau_area=0.05,
  ) -> dict[str, torch.Tensor]
  ```
  Returns `{"loss", "fidelity", "temporal", "risk", "soft_area"}`, all scalars.

- [ ] **Step 1: Write the failing tests**

Append to `training/tests/test_mitigator_losses.py`:

```python
def _profile():
    from detector.profiles import ThresholdProfile
    return ThresholdProfile(
        name="test", max_flashes_per_second=3, max_area_ratio=0.25,
        general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
        red_saturation_ratio_threshold=0.80,
    )


def _flickering_pair(seed=0):
    """Input frames: a dark frame and a much brighter one. Pure flicker."""
    g = torch.Generator().manual_seed(seed)
    dark = torch.rand(1, 3, 32, 32, generator=g) * 0.2
    bright = (dark + 0.7).clamp(0, 1)
    return dark, bright


def test_passing_the_input_through_unchanged_costs_zero_fidelity():
    from training.mitigator_losses import self_supervised_loss

    in_a, in_b = _flickering_pair()
    terms = self_supervised_loss(in_a, in_b, in_a, in_b, torch.zeros(1), torch.zeros(1), _profile())
    assert terms["fidelity"].item() == pytest.approx(0.0, abs=1e-6)


def test_the_do_nothing_output_is_still_charged_for_risk():
    # This is the whole design: passing the input straight through is free
    # on fidelity and free on nothing else. If the risk term did not fire
    # here, the model's optimum would be to do nothing.
    from training.mitigator_losses import self_supervised_loss

    in_a, in_b = _flickering_pair()
    terms = self_supervised_loss(in_a, in_b, in_a, in_b, torch.zeros(1), torch.zeros(1), _profile())
    assert terms["risk"].item() > 0.05
    assert terms["temporal"].item() > 0.05
    assert terms["loss"].item() > terms["fidelity"].item()


def test_flattening_the_pair_clears_the_risk_term_but_costs_fidelity():
    from training.mitigator_losses import self_supervised_loss

    in_a, in_b = _flickering_pair()
    flat = (in_a + in_b) / 2
    terms = self_supervised_loss(flat, flat, in_a, in_b, torch.zeros(1), torch.zeros(1), _profile())

    assert terms["risk"].item() < 0.01
    assert terms["temporal"].item() < 0.01
    assert terms["fidelity"].item() > 0.05


def test_lambda_risk_zero_removes_the_risk_term_entirely():
    # The pre-design objective, for regression comparison.
    from training.mitigator_losses import self_supervised_loss

    in_a, in_b = _flickering_pair()
    terms = self_supervised_loss(in_a, in_b, in_a, in_b, torch.zeros(1), torch.zeros(1),
                                 _profile(), lambda_risk=0.0)
    assert terms["loss"].item() == pytest.approx(
        (terms["fidelity"] + terms["temporal"]).item(), abs=1e-6
    )


def test_the_shift_is_applied_before_the_temporal_comparison():
    # A pure pan is not flicker. With the correct shift the two frames align
    # and the temporal term collapses; without it the same pair looks like a
    # large change. If this fails, the loss will fight camera movement.
    from training.mitigator_losses import self_supervised_loss

    g = torch.Generator().manual_seed(7)
    a = torch.rand(1, 3, 48, 48, generator=g) * 0.5 + 0.2
    b = torch.roll(a, shifts=4, dims=3)

    compensated = self_supervised_loss(a, b, a, b, torch.tensor([4.0]), torch.zeros(1), _profile())
    ignored = self_supervised_loss(a, b, a, b, torch.zeros(1), torch.zeros(1), _profile())
    assert compensated["temporal"].item() < ignored["temporal"].item() * 0.5


def test_all_terms_produce_finite_gradients():
    from training.mitigator_losses import self_supervised_loss

    in_a, in_b = _flickering_pair(seed=3)
    out_a = (in_a * 0.8).clone().requires_grad_(True)
    out_b = (in_b * 0.8).clone().requires_grad_(True)

    self_supervised_loss(out_a, out_b, in_a, in_b, torch.zeros(1), torch.zeros(1),
                         _profile())["loss"].backward()
    for grad in (out_a.grad, out_b.grad):
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


def test_reported_terms_sum_to_the_reported_loss():
    # The training loop logs these separately to watch the fidelity/risk
    # trade-off; a reported breakdown that does not reconstruct the total
    # would make that log misleading.
    from training.mitigator_losses import self_supervised_loss

    in_a, in_b = _flickering_pair(seed=5)
    lt, lr = 0.7, 1.3
    terms = self_supervised_loss(in_a * 0.9, in_b * 0.9, in_a, in_b,
                                 torch.zeros(1), torch.zeros(1), _profile(),
                                 lambda_temporal=lt, lambda_risk=lr)
    expected = terms["fidelity"] + lt * terms["temporal"] + lr * terms["risk"]
    assert terms["loss"].item() == pytest.approx(expected.item(), abs=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest training/tests/test_mitigator_losses.py -v`
Expected: FAIL — `ImportError: cannot import name 'self_supervised_loss'`

- [ ] **Step 3: Implement**

Append to `training/mitigator_losses.py`:

```python
def self_supervised_loss(
    out_a: torch.Tensor,
    out_b: torch.Tensor,
    in_a: torch.Tensor,
    in_b: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
    profile: "ThresholdProfile",
    lambda_temporal: float = 1.0,
    lambda_risk: float = 1.0,
    tau: float = 0.02,
    tau_area: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Loss for two consecutive frames, with no clean reference.

    Three terms pull against each other:

    - fidelity keeps the output on its input, so doing nothing is the
      default and every change has to be paid for;
    - temporal drives consecutive outputs together after cancelling camera
      motion, which is what actually removes flicker;
    - risk charges only for exceeding the Detector's own area limit, so the
      correction stops as soon as the clip is compliant instead of
      flattening everything a global weight happens to reach.

    The risk term is why this exists. Grading a frame against a clean
    reference, as the previous objective did, cannot see flicker at all --
    it is satisfied by correcting every frame in the same proportion, which
    is measurably what the trained model learned to do.

    `dx`/`dy` come from `detector.motion.estimate_global_shift` on the INPUT
    pair, not the outputs: the motion is a property of the footage, and
    using the Detector's own estimate keeps the loss scoring the alignment
    the pass/fail rule was computed on.
    """
    from training.soft_risk import risk_penalty, soft_flagged_area

    fidelity = F.l1_loss(out_a, in_a) + F.l1_loss(out_b, in_b)

    warped_out_a, valid = warp_by_shift(out_a, dx, dy)
    luminance_delta = (
        relative_luminance_torch(warped_out_a) - relative_luminance_torch(out_b)
    ).abs()
    denominator = valid.sum().clamp(min=1e-6)
    temporal = (luminance_delta * valid).sum() / denominator

    area = soft_flagged_area(warped_out_a, out_b, profile, tau=tau, valid_mask=valid)
    risk = risk_penalty(area, profile.max_area_ratio, tau_area=tau_area).mean()

    return {
        "loss": fidelity + lambda_temporal * temporal + lambda_risk * risk,
        "fidelity": fidelity,
        "temporal": temporal,
        "risk": risk,
        "soft_area": area.mean(),
    }
```

Add `from detector.profiles import ThresholdProfile` to the imports at the top of the file and drop the quotes from the annotation.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest training/tests/test_mitigator_losses.py -v`
Expected: PASS — existing tests plus 7 new ones.

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest -q`
Expected: PASS, 307 tests (284 + 4 + 7 + 5 + 7).

```bash
git add training/mitigator_losses.py training/tests/test_mitigator_losses.py
git commit -m "feat(training): self-supervised loss — fidelity, temporal consistency, risk penalty"
```

---

### Task 5: Verify the relaxation against the real Detector on real footage

The unit tests check the relaxation on synthetic pairs. This checks it where it will actually be used, and produces the number that decides whether Plan B is worth starting.

**Files:**
- Create: `scripts/soft_risk_report.py`
- Create: `docs/superpowers/specs/2026-08-07-soft-risk-agreement.md`

**Interfaces:**
- Consumes: `training.soft_risk.soft_flagged_area`, `detector.flash.transition_mask` / `red_flash_mask`, `detector.motion.estimate_global_shift` / `compensate_shift`, `scripts.screen_clean_clips.read_frames_lenient`.
- Produces: no code interface — a script and a measurement document.

- [ ] **Step 1: Write the script**

Create `scripts/soft_risk_report.py`:

```python
"""How closely the differentiable relaxation tracks the Detector it stands
in for, measured on real footage rather than synthetic pairs.

`training.soft_risk.soft_flagged_area` exists to put the pass/fail rule
inside the loss. If it disagrees with `detector.flash` on real frames, the
loss optimises something adjacent to the rule instead of the rule -- which
is the exact failure the self-supervised design was written to remove, so
this disagreement is worth measuring before anything is trained on it.

Frames are streamed, never accumulated: these clips reach several GB as
float32 and the tool exists to be run on them.

    python -m scripts.soft_risk_report --video clip.mp4 \\
        --profile configs/profiles/itu.json
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from detector.flash import red_flash_mask, transition_mask
from detector.luminance import relative_luminance
from detector.motion import compensate_shift, estimate_global_shift
from detector.profiles import load_profile
from training.soft_risk import soft_flagged_area


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare soft_flagged_area against the Detector's own area.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--max-dim", type=int, default=480)
    args = parser.parse_args(argv)

    profile = load_profile(Path(args.profile))
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")

    prev_rgb = prev_luminance = None
    hard_areas, soft_areas = [], []
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            h, w = frame_bgr.shape[:2]
            scale = args.max_dim / max(h, w)
            if scale < 1.0:
                frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)),
                                       interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            luminance = relative_luminance(frame_rgb)

            if prev_rgb is not None:
                dx, dy = estimate_global_shift(prev_luminance, luminance)
                aligned_rgb = compensate_shift(prev_rgb, dx, dy)
                aligned_luminance = compensate_shift(prev_luminance, dx, dy)

                hard = transition_mask(
                    aligned_luminance, luminance,
                    dark_threshold=profile.general_flash_dark_threshold,
                    delta_threshold=profile.general_flash_delta_threshold,
                ) | red_flash_mask(
                    aligned_rgb, frame_rgb,
                    saturation_ratio_threshold=profile.red_saturation_ratio_threshold,
                )
                hard_areas.append(float(hard.mean()))

                to_t = lambda f: torch.from_numpy(f.transpose(2, 0, 1)).float()[None]
                with torch.no_grad():
                    soft_areas.append(
                        float(soft_flagged_area(to_t(aligned_rgb), to_t(frame_rgb),
                                                profile, tau=args.tau).item())
                    )
            prev_rgb, prev_luminance = frame_rgb, luminance
    finally:
        capture.release()

    if not hard_areas:
        raise SystemExit(f"{args.video}: fewer than two frames decoded")

    hard = np.array(hard_areas)
    soft = np.array(soft_areas)
    error = np.abs(soft - hard)
    limit = profile.max_area_ratio
    agree = ((soft > limit) == (hard > limit)).mean()

    print(f"{Path(args.video).name}  {len(hard)} pairs  tau={args.tau}")
    print(f"  mean |soft - hard|     {error.mean():.4f}")
    print(f"  p95  |soft - hard|     {np.percentile(error, 95):.4f}")
    print(f"  max  |soft - hard|     {error.max():.4f}")
    print(f"  verdict agreement      {agree:.1%}  (both sides of area > {limit})")
    print(f"  hard mean / soft mean  {hard.mean():.4f} / {soft.mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it on the six evaluation clips**

```bash
python -m scripts.soft_risk_report --video test_mp4/wonbon_640.mp4 --profile configs/profiles/itu.json
python -m scripts.soft_risk_report --video test_mp4/anyma_640.mp4 --profile configs/profiles/itu.json
python -m scripts.soft_risk_report --video test_mp4/cera_640.mp4 --profile configs/profiles/itu.json
python -m scripts.soft_risk_report --video "data/raw/real_flicker_eval/videoplayback.mp4" --profile configs/profiles/itu.json
python -m scripts.soft_risk_report --video "data/raw/real_flicker_eval/16046-269122203_medium.mp4" --profile configs/profiles/itu.json
python -m scripts.soft_risk_report --video "data/raw/real_flicker_eval/1257-144566582_medium.mp4" --profile configs/profiles/itu.json
```

Run them one at a time; this machine has about 9 GB free and these clips are large.

- [ ] **Step 3: If agreement is poor, sweep tau before concluding**

If verdict agreement is below 95% on any clip, re-run that clip at `--tau 0.01` and `--tau 0.05` and record all three. `tau` is the width of the relaxation: too large and the sigmoid no longer resembles the threshold it replaces, too small and it becomes a step with no gradient. The spec's default of 0.02 was chosen as one fifth of `general_flash_delta_threshold` and has not been validated against real footage until now.

- [ ] **Step 4: Write the measurement document**

Create `docs/superpowers/specs/2026-08-07-soft-risk-agreement.md` with the real numbers from Steps 2 and 3 — no placeholders. It must contain:

- A table with one row per clip: pair count, mean / p95 / max `|soft − hard|`, verdict agreement, and the hard and soft means.
- The `tau` used, and the sweep results for any clip that needed one.
- A one-paragraph verdict stating plainly whether the relaxation tracks the Detector well enough to train against, and naming the worst clip and its number rather than only the average.
- If any clip's agreement is below 90%, say so and state what it implies for Plan B — the risk term would be steering toward a criterion the Detector does not share on that footage, and Plan B should not start until that is understood.

- [ ] **Step 5: Commit**

```bash
git add scripts/soft_risk_report.py docs/superpowers/specs/2026-08-07-soft-risk-agreement.md
git commit -m "test(training): measure the soft risk relaxation against the real Detector"
```

---

## Self-Review

**Spec coverage (sections 3 and 8 only — sections 4–7 are Plan B):**

| Spec requirement | Task |
|---|---|
| 3.1 `L_fid = mean\|out − in\|` | Task 4 |
| 3.2 temporal term, global shift, valid pixels | Tasks 3, 4 |
| 3.3 soft general flash | Task 2 |
| 3.3 soft red flash as \|difference of soft indicators\| | Task 2 |
| 3.3 `max` combining, softplus penalty | Task 2 |
| 3.3 frequency deliberately not relaxed | Task 2 (documented in the module docstring) |
| 3.4 defaults λ_t=1.0, λ_r=1.0, τ=0.02, τ_a=0.05 | Tasks 2, 4 signatures |
| 8 confidence guard, threshold from data | Task 1 |
| 8 "τ has not been validated on real footage" | Task 5 |

Sections 4, 5, 6, 7 (PriorCalc, model input, dataset/training loop, evaluation) are Plan B and deliberately absent.

**Type consistency:** `soft_flagged_area(prev_rgb, curr_rgb, profile, tau, valid_mask)` and `risk_penalty(area, max_area_ratio, tau_area)` are used with those exact signatures in Task 4 and Task 5. `warp_by_shift(frames, dx, dy) -> (warped, valid)` matches between Tasks 3 and 4. `estimate_global_shift(prev, curr, min_response)` matches between Tasks 1 and 5.

**One thing Task 4 assumes and Plan B must supply:** `dx`/`dy` are per-batch-item tensors computed from the input pair by `estimate_global_shift`. That is a NumPy/OpenCV call, so it belongs in the dataset (computed once, cached) rather than the training loop. Plan B's dataset task owns it.
