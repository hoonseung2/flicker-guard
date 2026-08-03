# Mitigator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Mitigator — a lightweight neural network that restores Detector-flagged risky frames toward their pre-flicker appearance, conditioned on PriorCalc's per-frame target illumination histogram and risk mask, plus the training pipeline that produces it from DatasetSynth's `data/synthetic/` output.

**Architecture:** `mitigator/arch.py` defines `MitigatorNet` — a small U-Net (3 downsample/upsample stages, self-attention bottleneck) that takes a 3-frame degraded window (t-1, t, t+1) plus a per-pixel risk mask and a 64-bin target histogram, and predicts a **residual** correction for frame t (not the frame itself). The caller adds the residual to the degraded frame, then blends the result with the mask so that pixels outside the detected risk region are always exactly the original input, regardless of what the network predicts. `training/mitigator_dataset.py` reads `data/synthetic/<sample_id>/` directories (written by DatasetSynth), computes-and-caches each sample's `FramePrior` sequence via `prior.compute.compute_prior`, and exposes a `torch.utils.data.Dataset` yielding one training example per risky, prior-conditioned frame. `training/train_mitigator.py` is an AdamW training loop with checkpoint/resume support. `mitigator/infer.py` is a minimal wrapper that runs a trained checkpoint over a sequence of frames, for evaluation purposes only (no runtime integration).

**Tech Stack:** Python 3.10+, numpy, opencv-python (already in requirements.txt), PyTorch 2.5+ (new dependency, this plan adds it to `requirements.txt`; already installed locally with CUDA support). No `torchmetrics` or `scikit-image` — SSIM is implemented by hand as a small differentiable function so it can be used directly as a training loss (not just an eval-only metric).

## Global Constraints

- **No Verifier/Fallback/BufferManager integration in this plan** — per the design spec, that is a separate future roadmap step. `mitigator/infer.py`'s `mitigate_segment` is standalone, used only for manual evaluation.
- **No production-scale training in this plan.** The automated test suite (`pytest`) must run entirely on tiny in-memory / `tmp_path` synthetic data, on CPU, in seconds — never on real `data/synthetic/` content and never assuming a GPU is present. Running a real multi-epoch training job against the actual 170-sample dataset is a manual step performed *after* this plan's final task, not part of any automated test.
- **Residual prediction + mask blending is not optional.** Every path that produces a "restored" frame — training loss computation and `mitigator/infer.py` alike — must compute `restored = degraded_center + model_output` and then `final = mask * restored + (1 - mask) * degraded_center`. This is the core safety guarantee from the design spec: pixels outside the detected risk mask are bit-for-bit identical to the input, never touched by the network's mistakes.
- **`target_histogram=None` frames are excluded from training and pass through unchanged at inference** — per `prior/compute.py`'s documented behavior (frames before Detector's trailing window fills, or clips with no clean frame at all, have no target). Never fabricate a substitute target.
- **`training/dataset_writer.py`'s `write_sample` gains a new required `fps: float` parameter** (not defaulted) — matching the project's existing precedent for `injection_mode` (added as required in the realistic-injection plan's fix wave) rather than silently defaulting, per that same reasoning: a missing/wrong value should fail loudly, not produce silently-wrong cached priors later.
- **`mitigator/weights/` is added to `.gitignore`** — trained checkpoints are large, regenerable binary artifacts, same treatment as `data/`.
- Every new file follows the existing project style: no comments explaining *what* code does, only *why* where genuinely non-obvious; module docstring at the top explaining purpose; tests colocated in a `tests/` subpackage.

---

## File Structure

```
flicker-guard/
├── requirements.txt              # Task 2 (add torch)
├── .gitignore                    # Task 6 (add mitigator/weights/)
├── mitigator/
│   ├── __init__.py               # Task 7
│   ├── arch.py                   # Task 2 — MitigatorNet, mitigate_frame
│   ├── infer.py                  # Task 6 — mitigate_segment
│   ├── weights/
│   │   └── .gitkeep              # Task 6
│   └── tests/
│       ├── __init__.py           # Task 7
│       ├── test_arch.py          # Task 2
│       └── test_infer.py         # Task 6
└── training/
    ├── dataset_writer.py         # Task 1 (modified — additive fps field)
    ├── cli.py                    # Task 1 (modified — one call-site update)
    ├── mitigator_losses.py       # Task 3 — ssim, psnr, l1_ssim_loss
    ├── mitigator_dataset.py      # Task 4 — clip_split, MitigatorDataset
    ├── train_mitigator.py        # Task 5 — training loop + CLI
    └── tests/
        ├── test_dataset_writer.py     # Task 1 (modified)
        ├── test_mitigator_losses.py   # Task 3
        ├── test_mitigator_dataset.py  # Task 4
        └── test_train_mitigator.py    # Task 5
```

---

## Task 1: Add `fps` to `write_sample`/`meta.json`

**Files:**
- Modify: `training/dataset_writer.py`
- Modify: `training/cli.py`
- Test: `training/tests/test_dataset_writer.py` (add one test, update 5 existing calls)

**Interfaces:**
- Consumes: nothing new.
- Produces: `write_sample(sample: SynthesizedSample, out_root: Path, clip_id: str, index: int, injection_mode: str, fps: float) -> str` — same as today plus one new required keyword parameter, recorded in `meta.json` as `"fps"`. Needed by Task 4's `training/mitigator_dataset.py`, which must know each clip's fps to re-run `compute_prior` and cannot reliably recover it from `injected_window.ramp_frames` (that field is `round(fps)`, a lossy integer rounding).

- [ ] **Step 1: Write the failing test**

Add to `training/tests/test_dataset_writer.py` (do not touch anything else yet):

```python
# add to training/tests/test_dataset_writer.py
def test_write_sample_records_fps(tmp_path):
    sid = write_sample(_sample(), tmp_path, clip_id="bear", index=0, injection_mode="flat", fps=24.0)
    meta = json.loads((tmp_path / sid / "meta.json").read_text(encoding="utf-8"))
    assert meta["fps"] == 24.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest training/tests/test_dataset_writer.py::test_write_sample_records_fps -v`
Expected: FAIL with `TypeError: write_sample() got an unexpected keyword argument 'fps'`

- [ ] **Step 3: Implement — add the `fps` parameter and fix existing call sites**

In `training/dataset_writer.py`, change:

```python
def write_sample(
    sample: SynthesizedSample,
    out_root: Path,
    clip_id: str,
    index: int,
    injection_mode: str,
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
        "injection_mode": injection_mode,
        "injected_window": dataclasses.asdict(sample.window),
        "segments": [dataclasses.asdict(s) for s in sample.segments],
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return sid
```

to:

```python
def write_sample(
    sample: SynthesizedSample,
    out_root: Path,
    clip_id: str,
    index: int,
    injection_mode: str,
    fps: float,
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
        "injection_mode": injection_mode,
        "fps": fps,
        "injected_window": dataclasses.asdict(sample.window),
        "segments": [dataclasses.asdict(s) for s in sample.segments],
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return sid
```

In `training/cli.py`, change the one call site:

```python
                    write_sample(sample, out_root, clip_id, index, injection_mode=injection_mode)
```

to:

```python
                    write_sample(sample, out_root, clip_id, index, injection_mode=injection_mode, fps=fps)
```

(`fps` is already a local variable in `run_batch` at that point, from `frame_iter, fps = read_video_frames(str(clip_path))` earlier in the same loop.)

In `training/tests/test_dataset_writer.py`, update every existing `write_sample(...)` call to also pass `fps=24.0`. There are 5 of them — the full updated file is:

```python
import json

import numpy as np
import pytest

import training.dataset_writer as dataset_writer
from detector.segments import RiskSegment
from training.dataset_writer import (
    sample_exists,
    sample_id,
    sample_injection_mode,
    write_sample,
)
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
    sid = write_sample(_sample(), tmp_path, clip_id="bear", index=0, injection_mode="flat", fps=24.0)
    sample_dir = tmp_path / sid
    assert (sample_dir / "clean" / "000000.png").exists()
    assert (sample_dir / "clean" / "000011.png").exists()
    assert (sample_dir / "degraded" / "000000.png").exists()
    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["clip_id"] == "bear"
    assert meta["profile"] == "kr"
    assert meta["pattern"] == "general"
    assert meta["injection_mode"] == "flat"
    assert meta["injected_window"]["start_frame"] == 1
    assert meta["segments"] == [{"start_frame": 3, "end_frame": 10}]


def test_write_sample_records_realistic_injection_mode(tmp_path):
    sid = write_sample(_sample(), tmp_path, clip_id="bear", index=0, injection_mode="realistic", fps=24.0)
    meta = json.loads((tmp_path / sid / "meta.json").read_text(encoding="utf-8"))
    assert meta["injection_mode"] == "realistic"


def test_write_sample_records_fps(tmp_path):
    sid = write_sample(_sample(), tmp_path, clip_id="bear", index=0, injection_mode="flat", fps=24.0)
    meta = json.loads((tmp_path / sid / "meta.json").read_text(encoding="utf-8"))
    assert meta["fps"] == 24.0


def test_sample_exists_true_after_write_false_before(tmp_path):
    sid = sample_id("bear", "kr", "general", 0)
    assert not sample_exists(tmp_path, sid)
    write_sample(_sample(), tmp_path, clip_id="bear", index=0, injection_mode="flat", fps=24.0)
    assert sample_exists(tmp_path, sid)


def test_sample_injection_mode_reads_back_recorded_mode(tmp_path):
    sid = write_sample(_sample(), tmp_path, clip_id="bear", index=0, injection_mode="realistic", fps=24.0)
    assert sample_injection_mode(tmp_path, sid) == "realistic"


def test_sample_injection_mode_defaults_to_flat_when_field_absent(tmp_path):
    # Samples written before this field existed have no "injection_mode" key
    # in meta.json. They were all produced by the (then only) flat path.
    sid = sample_id("bear", "kr", "general", 0)
    sample_dir = tmp_path / sid
    sample_dir.mkdir(parents=True)
    (sample_dir / "meta.json").write_text(json.dumps({"clip_id": "bear"}), encoding="utf-8")
    assert sample_injection_mode(tmp_path, sid) == "flat"


def test_write_sample_raises_and_writes_no_meta_when_a_frame_write_fails(tmp_path, monkeypatch):
    # cv2.imwrite returns False (it does not raise) on disk-full / permission
    # / path-length failures. If meta.json were still written, sample_exists
    # would report the truncated sample as complete and every future resume
    # run would skip it forever.
    monkeypatch.setattr(dataset_writer.cv2, "imwrite", lambda *args, **kwargs: False)

    with pytest.raises(OSError):
        write_sample(_sample(), tmp_path, clip_id="bear", index=0, injection_mode="flat", fps=24.0)

    sid = sample_id("bear", "kr", "general", 0)
    assert not (tmp_path / sid / "meta.json").exists()
    assert not sample_exists(tmp_path, sid)
```

- [ ] **Step 4: Run tests to verify everything passes**

Run: `pytest training/tests/test_dataset_writer.py training/tests/test_cli.py -v`
Expected: all PASS (no regressions in `test_cli.py`, which exercises `run_batch` end-to-end through the updated call site)

- [ ] **Step 5: Commit**

```bash
git add training/dataset_writer.py training/cli.py training/tests/test_dataset_writer.py
git commit -m "feat(mitigator): record fps in sample metadata"
```

---

## Task 2: `mitigator/arch.py` — `MitigatorNet` and `mitigate_frame`

**Files:**
- Create: `mitigator/__init__.py` (empty, package marker — needed now so `mitigator.arch` is importable; the rest of package setup happens in Task 7)
- Create: `mitigator/arch.py`
- Create: `mitigator/tests/__init__.py` (empty)
- Test: `mitigator/tests/test_arch.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: nothing from this project — pure PyTorch.
- Produces: `MitigatorNet` (an `nn.Module` with `forward(window: Tensor[B,9,H,W], mask: Tensor[B,1,H,W], histogram: Tensor[B,64]) -> Tensor[B,3,H,W]`, predicting a residual, not a full frame). `mitigate_frame(window: Tensor[B,9,H,W], mask: Tensor[B,1,H,W], histogram: Tensor[B,64], model: MitigatorNet) -> Tensor[B,3,H,W]` — applies the model and the mask-blend safety guarantee. Both are consumed by Task 3's loss tests, Task 5's training loop, and Task 6's `infer.py`.

- [ ] **Step 1: Add the dependency**

In `requirements.txt`, add a line:

```
torch>=2.5
```

(full file becomes:)

```
numpy>=1.24
opencv-python>=4.8
pytest>=7.4
torch>=2.5
```

- [ ] **Step 2: Write the failing tests**

Create `mitigator/__init__.py` (empty file) and `mitigator/tests/__init__.py` (empty file).

Create `mitigator/tests/test_arch.py`:

```python
import torch

from mitigator.arch import MitigatorNet, mitigate_frame


def test_forward_output_shape_matches_input_resolution():
    model = MitigatorNet()
    window = torch.rand(2, 9, 32, 32)
    mask = torch.rand(2, 1, 32, 32)
    histogram = torch.rand(2, 64)
    out = model(window, mask, histogram)
    assert out.shape == (2, 3, 32, 32)


def test_forward_output_shape_with_resolution_not_a_multiple_of_8():
    # Real inference runs on arbitrary clip resolutions (e.g. 854x480, not
    # divisible by 8) -- the encoder/decoder's downsample-then-upsample must
    # not silently change the output size.
    model = MitigatorNet()
    window = torch.rand(1, 9, 37, 45)
    mask = torch.rand(1, 1, 37, 45)
    histogram = torch.rand(1, 64)
    out = model(window, mask, histogram)
    assert out.shape == (1, 3, 37, 45)


def test_gradients_flow_to_every_parameter():
    model = MitigatorNet()
    window = torch.rand(1, 9, 32, 32)
    mask = torch.rand(1, 1, 32, 32)
    histogram = torch.rand(1, 64)
    out = model(window, mask, histogram)
    out.sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient reached parameter {name}"


def test_mitigate_frame_preserves_pixels_outside_mask_exactly():
    # The core safety guarantee: wherever mask == 0, the output must be
    # bit-for-bit the input, regardless of what an untrained (effectively
    # random) network predicts.
    model = MitigatorNet()
    model.eval()
    window = torch.rand(1, 9, 16, 16)
    mask = torch.zeros(1, 1, 16, 16)
    histogram = torch.rand(1, 64)
    with torch.no_grad():
        result = mitigate_frame(window, mask, histogram, model)
    center = window[:, 3:6, :, :]
    assert torch.equal(result, center)


def test_mitigate_frame_returns_center_frame_shape():
    model = MitigatorNet()
    model.eval()
    window = torch.rand(1, 9, 20, 24)
    mask = torch.ones(1, 1, 20, 24)
    histogram = torch.rand(1, 64)
    with torch.no_grad():
        result = mitigate_frame(window, mask, histogram, model)
    assert result.shape == (1, 3, 20, 24)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest mitigator/tests/test_arch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mitigator.arch'`

- [ ] **Step 4: Implement `mitigator/arch.py`**

```python
"""MitigatorNet: predicts a residual correction for the center frame of a
3-frame degraded window, conditioned on a per-pixel risk mask and a target
illumination histogram (both from prior.compute.compute_prior).

Residual, not full-frame, prediction: degraded and clean frames are
identical outside the injected/risky region by construction (DatasetSynth
only ever modifies pixels inside the mask), so predicting "what changed"
is an easier target than reconstructing the whole frame from scratch, and
naturally biases the untrained network toward leaving already-clean pixels
alone. mitigate_frame's mask blend then makes that bias a hard guarantee
rather than a hope: pixels outside the mask are the original input,
unconditionally, no matter what the network outputs.

Architecture is a small U-Net with a self-attention bottleneck -- inspired
by Flickerformer's "3-frame window -> lightweight Restormer-style network"
idea (README section 1), reimplemented from scratch rather than adapted
from Flickerformer's published code, which ships with no LICENSE file.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_HIST_BINS = 64
_HIST_EMBED_CHANNELS = 16


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = _ConvBlock(in_channels, out_channels)
        self.pool = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.block(x)
        down = self.pool(skip)
        return down, skip


class _Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.block = _ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            # Odd input resolutions round differently on the way down vs
            # back up (e.g. 37 -> 19 -> 37 is fine, but 37 -> 19 -> 38 is
            # not) -- nudge back to the skip connection's exact size rather
            # than fail on non-power-of-2-friendly resolutions.
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class _BottleneckAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        normed = self.norm(x)
        tokens = normed.flatten(2).transpose(1, 2)  # (B, H*W, C)
        attended, _ = self.attn(tokens, tokens, tokens)
        attended = attended.transpose(1, 2).reshape(b, c, h, w)
        return x + attended


class MitigatorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hist_embed = nn.Sequential(
            nn.Linear(_HIST_BINS, 32),
            nn.ReLU(),
            nn.Linear(32, _HIST_EMBED_CHANNELS),
        )
        in_channels = 3 * 3 + 1 + _HIST_EMBED_CHANNELS  # 3 RGB frames + mask + hist embed
        self.down1 = _Down(in_channels, 32)
        self.down2 = _Down(32, 64)
        self.down3 = _Down(64, 128)
        self.bottleneck = _BottleneckAttention(128)
        self.up3 = _Up(128, 128, 64)
        self.up2 = _Up(64, 64, 32)
        self.up1 = _Up(32, 32, 32)
        self.out_conv = nn.Conv2d(32, 3, 3, padding=1)

    def forward(
        self,
        window: torch.Tensor,
        mask: torch.Tensor,
        histogram: torch.Tensor,
    ) -> torch.Tensor:
        b, _, h, w = window.shape
        hist_embedding = self.hist_embed(histogram)
        hist_spatial = hist_embedding.view(b, _HIST_EMBED_CHANNELS, 1, 1).expand(-1, -1, h, w)
        x = torch.cat([window, mask, hist_spatial], dim=1)

        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        x3 = self.bottleneck(x3)
        x = self.up3(x3, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        return self.out_conv(x)


def mitigate_frame(
    window: torch.Tensor,
    mask: torch.Tensor,
    histogram: torch.Tensor,
    model: MitigatorNet,
) -> torch.Tensor:
    delta = model(window, mask, histogram)
    center = window[:, 3:6, :, :]
    restored = center + delta
    return mask * restored + (1 - mask) * center
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest mitigator/tests/test_arch.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt mitigator/__init__.py mitigator/arch.py mitigator/tests/__init__.py mitigator/tests/test_arch.py
git commit -m "feat(mitigator): add MitigatorNet architecture and mask-blend safety wrapper"
```

---

## Task 3: `training/mitigator_losses.py` — differentiable SSIM, PSNR, combined loss

**Files:**
- Create: `training/mitigator_losses.py`
- Test: `training/tests/test_mitigator_losses.py`

**Interfaces:**
- Consumes: nothing from this project — pure PyTorch.
- Produces: `ssim(img1: Tensor[B,C,H,W], img2: Tensor[B,C,H,W], window_size: int = 11) -> Tensor[scalar]`, `psnr(pred: Tensor, target: Tensor) -> Tensor[scalar]`, `l1_ssim_loss(pred: Tensor, target: Tensor, lambda_ssim: float = 0.5) -> Tensor[scalar]`. Consumed by Task 5's training loop.

- [ ] **Step 1: Write the failing tests**

Create `training/tests/test_mitigator_losses.py`:

```python
import torch

from training.mitigator_losses import l1_ssim_loss, psnr, ssim


def test_ssim_of_identical_images_is_one():
    img = torch.rand(1, 3, 32, 32)
    assert torch.isclose(ssim(img, img), torch.tensor(1.0), atol=1e-4)


def test_ssim_of_very_different_images_is_lower_than_identical():
    img1 = torch.zeros(1, 3, 32, 32)
    img2 = torch.ones(1, 3, 32, 32)
    assert ssim(img1, img2) < ssim(img1, img1)


def test_psnr_of_identical_images_is_very_high():
    img = torch.rand(1, 3, 16, 16)
    assert psnr(img, img) > 40.0


def test_psnr_decreases_as_images_diverge():
    img1 = torch.full((1, 3, 16, 16), 0.5)
    img2_close = img1 + 0.01
    img2_far = img1 + 0.4
    assert psnr(img1, img2_close) > psnr(img1, img2_far)


def test_l1_ssim_loss_is_near_zero_for_identical_images():
    img = torch.rand(1, 3, 16, 16)
    loss = l1_ssim_loss(img, img)
    assert loss.item() < 1e-4


def test_l1_ssim_loss_is_positive_for_different_images():
    img1 = torch.zeros(1, 3, 16, 16)
    img2 = torch.ones(1, 3, 16, 16)
    loss = l1_ssim_loss(img1, img2)
    assert loss.item() > 0.0


def test_l1_ssim_loss_gradients_flow():
    pred = torch.rand(1, 3, 16, 16, requires_grad=True)
    target = torch.rand(1, 3, 16, 16)
    loss = l1_ssim_loss(pred, target)
    loss.backward()
    assert pred.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest training/tests/test_mitigator_losses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.mitigator_losses'`

- [ ] **Step 3: Implement `training/mitigator_losses.py`**

```python
"""Differentiable SSIM/PSNR and the combined L1+SSIM training loss for
Mitigator. Hand-implemented rather than pulled from torchmetrics: SSIM must
be differentiable to use as a training loss (not just an eval-time metric),
and the standard windowed formulation (Wang et al. 2004) is short enough
that adding a new dependency isn't worth it.
"""
import torch
import torch.nn.functional as F

_C1 = 0.01 ** 2
_C2 = 0.03 ** 2


def _gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _ssim_window(window_size: int, channels: int) -> torch.Tensor:
    g1d = _gaussian_kernel(window_size, sigma=1.5)
    g2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)
    return g2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Mean SSIM over the batch. Inputs are (B, C, H, W) float tensors,
    values expected in [0, 1]."""
    channels = img1.shape[1]
    window = _ssim_window(window_size, channels).to(img1.device, img1.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + _C1) * (2 * sigma12 + _C2)) / (
        (mu1_sq + mu2_sq + _C1) * (sigma1_sq + sigma2_sq + _C2)
    )
    return ssim_map.mean()


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    return 10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))


def l1_ssim_loss(pred: torch.Tensor, target: torch.Tensor, lambda_ssim: float = 0.5) -> torch.Tensor:
    l1 = F.l1_loss(pred, target)
    return l1 + lambda_ssim * (1.0 - ssim(pred, target))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest training/tests/test_mitigator_losses.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add training/mitigator_losses.py training/tests/test_mitigator_losses.py
git commit -m "feat(mitigator): add differentiable SSIM/PSNR and L1+SSIM training loss"
```

---

## Task 4: `training/mitigator_dataset.py` — data pipeline

**Files:**
- Create: `training/mitigator_dataset.py`
- Test: `training/tests/test_mitigator_dataset.py`

**Interfaces:**
- Consumes: `prior.compute.compute_prior(frames, fps, profile, n_bins=64) -> list[FramePrior]` and `prior.compute.FramePrior` (fields: `frame_index: int`, `target_histogram: np.ndarray | None`, `mask: np.ndarray`), `detector.profiles.ThresholdProfile`. Reads `data/synthetic/<sample_id>/{clean,degraded}/{i:06d}.png` and `meta.json` (`clip_id`, `fps`, `segments`: list of `{start_frame, end_frame}`) as written by Task 1's `write_sample`.
- Produces: `clip_split(clip_id: str) -> str` (`"train"` or `"val"`, deterministic), `MitigatorDataset(data_dir: Path, profile: ThresholdProfile, split: str, patch_size: int | None = None)` — a `torch.utils.data.Dataset` whose `__getitem__` returns a dict with keys `"window"` (Tensor `[9,H,W]`), `"mask"` (Tensor `[1,H,W]`), `"histogram"` (Tensor `[64]`), `"clean"` (Tensor `[3,H,W]`). Consumed by Task 5's training loop.

- [ ] **Step 1: Write the failing tests**

Create `training/tests/test_mitigator_dataset.py`:

```python
import json

import numpy as np
import pytest

from detector.profiles import ThresholdProfile
from training.dataset_writer import _write_frames
from training.mitigator_dataset import MitigatorDataset, clip_split

PROFILE = ThresholdProfile(
    name="test", max_flashes_per_second=3, max_area_ratio=0.10,
    general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
    red_saturation_ratio_threshold=0.80,
)


def _write_fake_sample(root, clip_id, n_frames=20, h=8, w=8, fps=10.0, segments=None):
    sample_dir = root / f"{clip_id}__test__general__000"
    frames = [np.full((h, w, 3), 0.5, dtype=np.float32) for _ in range(n_frames)]
    _write_frames(frames, sample_dir / "clean")
    _write_frames(frames, sample_dir / "degraded")
    meta = {
        "clip_id": clip_id,
        "fps": fps,
        "profile": "test",
        "pattern": "general",
        "injection_mode": "flat",
        "segments": segments if segments is not None else [],
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return sample_dir


def test_clip_split_is_deterministic():
    assert clip_split("bear") == clip_split("bear")


def test_clip_split_covers_both_train_and_val_across_many_ids():
    splits = {clip_split(f"clip_{i}") for i in range(50)}
    assert splits == {"train", "val"}


def test_dataset_excludes_frames_outside_segments(tmp_path):
    # frames 0-19 exist, but no segments -> nothing is "risky", so len == 0
    # regardless of what compute_prior finds.
    _write_fake_sample(tmp_path, "clipA", segments=[])
    split = clip_split("clipA")
    dataset = MitigatorDataset(tmp_path, PROFILE, split=split)
    assert len(dataset) == 0


def test_dataset_includes_only_frames_in_segments_with_valid_target(tmp_path):
    # fps=10 -> Detector's trailing window needs ~10 frames before it stops
    # reporting uncertain=True, so target_histogram is None for frames
    # 0-9ish on any clip. Frames 12-15 are past that priming window.
    _write_fake_sample(tmp_path, "clipB", n_frames=20, fps=10.0,
                       segments=[{"start_frame": 12, "end_frame": 15}])
    split = clip_split("clipB")
    dataset = MitigatorDataset(tmp_path, PROFILE, split=split)
    assert len(dataset) == 4


def test_getitem_returns_correct_shapes(tmp_path):
    _write_fake_sample(tmp_path, "clipC", n_frames=20, h=8, w=8, fps=10.0,
                       segments=[{"start_frame": 12, "end_frame": 15}])
    split = clip_split("clipC")
    dataset = MitigatorDataset(tmp_path, PROFILE, split=split)
    example = dataset[0]
    assert example["window"].shape == (9, 8, 8)
    assert example["mask"].shape == (1, 8, 8)
    assert example["histogram"].shape == (64,)
    assert example["clean"].shape == (3, 8, 8)


def test_getitem_duplicates_boundary_frame_at_clip_end(tmp_path):
    # segment reaches the last frame index (19 in a 20-frame clip) -- the
    # window's "next" frame has no real neighbor and must duplicate the
    # center frame instead of indexing out of range.
    _write_fake_sample(tmp_path, "clipD", n_frames=20, h=8, w=8, fps=10.0,
                       segments=[{"start_frame": 12, "end_frame": 19}])
    split = clip_split("clipD")
    dataset = MitigatorDataset(tmp_path, PROFILE, split=split)
    last_examples = [ex for ex in dataset if True]
    # find the example for frame_index 19 by checking dataset._index directly
    frame_indices = [idx for _, idx in dataset._index]
    assert 19 in frame_indices
    pos = frame_indices.index(19)
    example = dataset[pos]
    center = example["window"][3:6, :, :]
    next_frame = example["window"][6:9, :, :]
    assert torch.equal(center, next_frame)


def test_prior_cache_is_written_and_reused(tmp_path, monkeypatch):
    _write_fake_sample(tmp_path, "clipE", n_frames=20, fps=10.0,
                       segments=[{"start_frame": 12, "end_frame": 15}])
    split = clip_split("clipE")

    MitigatorDataset(tmp_path, PROFILE, split=split)
    sample_dir = tmp_path / "clipE__test__general__000"
    assert (sample_dir / "prior_cache.npz").exists()

    import training.mitigator_dataset as mitigator_dataset_module

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("compute_prior should not be called when a cache file exists")

    monkeypatch.setattr(mitigator_dataset_module, "compute_prior", _fail_if_called)
    MitigatorDataset(tmp_path, PROFILE, split=split)  # must not raise
```

Add `import torch` to the top of `training/tests/test_mitigator_dataset.py` alongside the other imports (needed for `torch.equal` in `test_getitem_duplicates_boundary_frame_at_clip_end`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest training/tests/test_mitigator_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.mitigator_dataset'`

- [ ] **Step 3: Implement `training/mitigator_dataset.py`**

```python
"""Bridges DatasetSynth's data/synthetic/ output and PriorCalc's
compute_prior into training examples for Mitigator: one example per risky,
prior-conditioned frame, as a (3-frame degraded window, risk mask, target
histogram) -> clean center-frame pair.

Train/val split is a deterministic hash of clip_id, not a stored list --
same philosophy as training.cli's per-sample rng derivation: the split is a
pure function of the input, so it never depends on scan order and never
needs a separate file kept in sync. All samples for a given clip (both
general and red pattern) land on the same side, so no clip's content leaks
across the split.

compute_prior's per-frame illumination histogram is expensive to reproduce
every epoch (it re-runs Detector's full classical pipeline, including
motion estimation, over the whole clip) -- it is computed once per sample
and cached to prior_cache.npz alongside that sample's frames, mirroring
DatasetSynth's own resume-by-checking-what-exists philosophy.
"""
import json
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from detector.profiles import ThresholdProfile
from prior.compute import FramePrior, compute_prior

_TRAIN_PCT = 80


def clip_split(clip_id: str) -> str:
    return "train" if zlib.crc32(clip_id.encode("utf-8")) % 100 < _TRAIN_PCT else "val"


def _read_frame(frames_dir: Path, index: int) -> np.ndarray:
    path = frames_dir / f"{index:06d}.png"
    bgr_uint8 = cv2.imread(str(path))
    if bgr_uint8 is None:
        raise OSError(f"failed to read frame: {path}")
    return cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _read_frame_sequence(frames_dir: Path) -> list[np.ndarray]:
    n = len(list(frames_dir.glob("*.png")))
    return [_read_frame(frames_dir, i) for i in range(n)]


def _save_prior_cache(cache_path: Path, priors: list[FramePrior]) -> None:
    n = len(priors)
    n_bins = next((p.target_histogram.shape[0] for p in priors if p.target_histogram is not None), 64)
    has_target = np.array([p.target_histogram is not None for p in priors], dtype=bool)
    histograms = np.zeros((n, n_bins), dtype=np.float32)
    for i, p in enumerate(priors):
        if p.target_histogram is not None:
            histograms[i] = p.target_histogram
    masks = np.stack([p.mask for p in priors]).astype(bool)
    frame_indices = np.array([p.frame_index for p in priors], dtype=np.int64)
    np.savez_compressed(
        cache_path,
        has_target=has_target,
        histograms=histograms,
        masks=masks,
        frame_indices=frame_indices,
    )


def _load_prior_cache(cache_path: Path) -> list[FramePrior]:
    data = np.load(cache_path)
    priors = []
    for i in range(len(data["frame_indices"])):
        target = data["histograms"][i] if data["has_target"][i] else None
        priors.append(FramePrior(
            frame_index=int(data["frame_indices"][i]),
            target_histogram=target,
            mask=data["masks"][i],
        ))
    return priors


def _load_or_compute_prior(
    sample_dir: Path, fps: float, profile: ThresholdProfile
) -> list[FramePrior]:
    cache_path = sample_dir / "prior_cache.npz"
    if cache_path.exists():
        return _load_prior_cache(cache_path)
    degraded_frames = _read_frame_sequence(sample_dir / "degraded")
    priors = compute_prior(degraded_frames, fps=fps, profile=profile)
    _save_prior_cache(cache_path, priors)
    return priors


class MitigatorDataset(Dataset):
    def __init__(self, data_dir: Path, profile: ThresholdProfile, split: str):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.data_dir = Path(data_dir)
        self._priors_by_sample: dict[Path, dict[int, FramePrior]] = {}
        self._index: list[tuple[Path, int]] = []

        for sample_dir in sorted(self.data_dir.iterdir()):
            meta_path = sample_dir / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if clip_split(meta["clip_id"]) != split:
                continue

            priors = _load_or_compute_prior(sample_dir, meta["fps"], profile)
            self._priors_by_sample[sample_dir] = {p.frame_index: p for p in priors}

            risky_indices: set[int] = set()
            for segment in meta["segments"]:
                risky_indices.update(range(segment["start_frame"], segment["end_frame"] + 1))

            for prior in priors:
                if prior.frame_index in risky_indices and prior.target_histogram is not None:
                    self._index.append((sample_dir, prior.frame_index))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        sample_dir, frame_index = self._index[idx]
        priors = self._priors_by_sample[sample_dir]
        prior = priors[frame_index]
        n_frames = len(priors)

        prev_idx = max(0, frame_index - 1)
        next_idx = min(n_frames - 1, frame_index + 1)
        degraded_dir = sample_dir / "degraded"
        window = np.concatenate(
            [
                _read_frame(degraded_dir, prev_idx),
                _read_frame(degraded_dir, frame_index),
                _read_frame(degraded_dir, next_idx),
            ],
            axis=-1,
        )
        clean_center = _read_frame(sample_dir / "clean", frame_index)

        return {
            "window": torch.from_numpy(window.transpose(2, 0, 1)).float(),
            "mask": torch.from_numpy(prior.mask[None, :, :].astype(np.float32)),
            "histogram": torch.from_numpy(prior.target_histogram).float(),
            "clean": torch.from_numpy(clean_center.transpose(2, 0, 1)).float(),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest training/tests/test_mitigator_dataset.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add training/mitigator_dataset.py training/tests/test_mitigator_dataset.py
git commit -m "feat(mitigator): add training data pipeline bridging DatasetSynth and PriorCalc"
```

---

## Task 5: `training/train_mitigator.py` — training loop and CLI

**Files:**
- Create: `training/train_mitigator.py`
- Test: `training/tests/test_train_mitigator.py`

**Interfaces:**
- Consumes: `mitigator.arch.MitigatorNet`, `mitigator.arch.mitigate_frame`, `training.mitigator_losses.l1_ssim_loss`, `training.mitigator_losses.psnr`, `training.mitigator_dataset.MitigatorDataset`, `detector.profiles.load_profile`.
- Produces: `save_checkpoint(path: Path, model: MitigatorNet, optimizer: torch.optim.Optimizer, epoch: int) -> None`, `load_checkpoint(path: Path, model: MitigatorNet, optimizer: torch.optim.Optimizer) -> int` (returns the saved epoch number), `train_one_epoch(model, optimizer, dataloader, device) -> float` (returns mean training loss), `evaluate(model, dataloader, device) -> dict` (returns `{"loss": float, "psnr": float}` on a val set), `main(argv: list[str] | None = None) -> int` (CLI entry point).

- [ ] **Step 1: Write the failing tests**

Create `training/tests/test_train_mitigator.py`:

```python
import torch

from mitigator.arch import MitigatorNet
from training.mitigator_losses import l1_ssim_loss
from training.train_mitigator import load_checkpoint, save_checkpoint, train_one_epoch


class _FixedBatchDataset(torch.utils.data.Dataset):
    """A dataset that always returns the same single example -- lets a
    training-loop test check "does loss go down" without needing real
    data/synthetic/ content or a GPU."""

    def __init__(self, h=16, w=16, n=4):
        torch.manual_seed(0)
        self.window = torch.rand(9, h, w)
        self.mask = torch.ones(1, h, w)
        self.histogram = torch.rand(64)
        self.clean = torch.rand(3, h, w)
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "window": self.window,
            "mask": self.mask,
            "histogram": self.histogram,
            "clean": self.clean,
        }


def test_train_one_epoch_reduces_loss_over_many_epochs():
    torch.manual_seed(0)
    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    dataloader = torch.utils.data.DataLoader(_FixedBatchDataset(), batch_size=4)

    first_epoch_loss = train_one_epoch(model, optimizer, dataloader, device="cpu")
    last_epoch_loss = first_epoch_loss
    for _ in range(19):
        last_epoch_loss = train_one_epoch(model, optimizer, dataloader, device="cpu")

    assert last_epoch_loss < first_epoch_loss


def test_train_one_epoch_returns_finite_loss():
    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataloader = torch.utils.data.DataLoader(_FixedBatchDataset(), batch_size=4)
    loss = train_one_epoch(model, optimizer, dataloader, device="cpu")
    assert loss == loss  # NaN check (NaN != NaN)
    assert loss != float("inf")


def test_checkpoint_save_and_load_restores_model_and_optimizer_state(tmp_path):
    torch.manual_seed(0)
    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataloader = torch.utils.data.DataLoader(_FixedBatchDataset(), batch_size=4)
    train_one_epoch(model, optimizer, dataloader, device="cpu")
    train_one_epoch(model, optimizer, dataloader, device="cpu")

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, optimizer, epoch=2)

    new_model = MitigatorNet()
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    restored_epoch = load_checkpoint(checkpoint_path, new_model, new_optimizer)

    assert restored_epoch == 2
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)

    orig_state = list(optimizer.state.values())[0]
    new_state = list(new_optimizer.state.values())[0]
    assert torch.equal(orig_state["exp_avg"], new_state["exp_avg"])


def test_train_one_epoch_raises_clearly_on_nan_loss(monkeypatch):
    # A NaN loss must never be swallowed -- it means something upstream
    # (bad data, exploding lr, etc.) is broken, and silently continuing
    # would corrupt every weight via backward() on NaN.
    import training.train_mitigator as train_mitigator_module

    def _nan_loss(pred, target):
        return pred.sum() * float("nan")

    monkeypatch.setattr(train_mitigator_module, "l1_ssim_loss", _nan_loss)

    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataloader = torch.utils.data.DataLoader(_FixedBatchDataset(), batch_size=4)

    with pytest.raises(RuntimeError, match="NaN"):
        train_one_epoch(model, optimizer, dataloader, device="cpu")
```

Add `import pytest` to the top of `training/tests/test_train_mitigator.py` alongside the `import torch` line (needed for `pytest.raises`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest training/tests/test_train_mitigator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.train_mitigator'`

- [ ] **Step 3: Implement `training/train_mitigator.py`**

```python
"""Training loop for Mitigator: AdamW + L1/SSIM loss over
training.mitigator_dataset.MitigatorDataset, with checkpoint/resume support
so a multi-hour run surviving an interruption doesn't have to restart from
scratch (same philosophy as DatasetSynth's resumable batch CLI).
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from detector.profiles import load_profile
from mitigator.arch import MitigatorNet, mitigate_frame
from training.mitigator_dataset import MitigatorDataset
from training.mitigator_losses import l1_ssim_loss, psnr


def save_checkpoint(path: Path, model: MitigatorNet, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
        path,
    )


def load_checkpoint(path: Path, model: MitigatorNet, optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["epoch"]


def train_one_epoch(
    model: MitigatorNet,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: str,
) -> float:
    model.train()
    model.to(device)
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        window = batch["window"].to(device)
        mask = batch["mask"].to(device)
        histogram = batch["histogram"].to(device)
        clean = batch["clean"].to(device)

        optimizer.zero_grad()
        restored = mitigate_frame(window, mask, histogram, model)
        loss = l1_ssim_loss(restored, clean)
        loss_value = loss.item()
        if loss_value != loss_value:  # NaN check (NaN != NaN) -- fail loud, never train through it
            raise RuntimeError(
                "train_one_epoch: loss is NaN. This is a signal of a real bug "
                "(bad data, exploding learning rate, etc.), not something to "
                "silently train through -- backward() on a NaN loss would "
                "corrupt every weight in the model."
            )
        loss.backward()
        optimizer.step()

        total_loss += loss_value
        n_batches += 1
    return total_loss / n_batches


def evaluate(model: MitigatorNet, dataloader: DataLoader, device: str) -> dict:
    model.eval()
    model.to(device)
    total_loss = 0.0
    total_psnr = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            window = batch["window"].to(device)
            mask = batch["mask"].to(device)
            histogram = batch["histogram"].to(device)
            clean = batch["clean"].to(device)

            restored = mitigate_frame(window, mask, histogram, model)
            total_loss += l1_ssim_loss(restored, clean).item()
            total_psnr += psnr(restored, clean).item()
            n_batches += 1
    return {"loss": total_loss / n_batches, "psnr": total_psnr / n_batches}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Mitigator on DatasetSynth output.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--profile", required=True, help="path to a profile JSON, e.g. configs/profiles/itu.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-path")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    profile = load_profile(Path(args.profile))
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path) if args.log_path else checkpoint_dir / "train_log.jsonl"

    train_dataset = MitigatorDataset(Path(args.data_dir), profile, split="train")
    val_dataset = MitigatorDataset(Path(args.data_dir), profile, split="val")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 0
    latest_path = checkpoint_dir / "latest.pt"
    if args.resume and latest_path.exists():
        start_epoch = load_checkpoint(latest_path, model, optimizer) + 1

    best_val_loss = float("inf")
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - start_time

        save_checkpoint(latest_path, model, optimizer, epoch)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_psnr": val_metrics["psnr"],
                "elapsed_seconds": elapsed,
            }) + "\n")
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
              f"val_psnr={val_metrics['psnr']:.2f} ({elapsed:.1f}s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest training/tests/test_train_mitigator.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add training/train_mitigator.py training/tests/test_train_mitigator.py
git commit -m "feat(mitigator): add training loop with checkpoint/resume support"
```

---

## Task 6: `mitigator/infer.py` — evaluation-only inference wrapper

**Files:**
- Create: `mitigator/infer.py`
- Test: `mitigator/tests/test_infer.py`
- Modify: `.gitignore`
- Create: `mitigator/weights/.gitkeep`

**Interfaces:**
- Consumes: `mitigator.arch.MitigatorNet`, `mitigator.arch.mitigate_frame`, `prior.compute.compute_prior`, `detector.profiles.ThresholdProfile`.
- Produces: `mitigate_segment(frames: list[np.ndarray], fps: float, profile: ThresholdProfile, model: MitigatorNet) -> list[np.ndarray]`. Not consumed by any other task in this plan — this is the evaluation-only entry point the design spec calls for (no Verifier/BufferManager wiring).

- [ ] **Step 1: Add `mitigator/weights/` to `.gitignore`**

Change `.gitignore` from:

```
__pycache__/
*.pyc
.pytest_cache/

# Raw and generated datasets — large binary content, never committed.
data/
```

to:

```
__pycache__/
*.pyc
.pytest_cache/

# Raw and generated datasets — large binary content, never committed.
data/

# Trained Mitigator checkpoints — large, regenerable binary artifacts.
mitigator/weights/*
!mitigator/weights/.gitkeep
```

Create an empty file `mitigator/weights/.gitkeep` so the directory itself is tracked even though its contents are ignored.

- [ ] **Step 2: Write the failing tests**

Create `mitigator/tests/test_infer.py`:

```python
import numpy as np
import torch

from detector.profiles import ThresholdProfile
from mitigator.arch import MitigatorNet
from mitigator.infer import mitigate_segment

PROFILE = ThresholdProfile(
    name="test", max_flashes_per_second=3, max_area_ratio=0.10,
    general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
    red_saturation_ratio_threshold=0.80,
)


def _flat_frames(n, h, w, value=0.5):
    return [np.full((h, w, 3), value, dtype=np.float32) for _ in range(n)]


def test_mitigate_segment_returns_same_length_and_shape():
    torch.manual_seed(0)
    model = MitigatorNet()
    frames = _flat_frames(20, 16, 16)
    result = mitigate_segment(frames, fps=10.0, profile=PROFILE, model=model)
    assert len(result) == len(frames)
    for out_frame, in_frame in zip(result, frames):
        assert out_frame.shape == in_frame.shape


def test_mitigate_segment_passes_through_frames_with_no_target_histogram():
    # fps=10 means the first ~10 frames of any clip have target_histogram
    # None (Detector's trailing window hasn't filled yet) -- those frames
    # must come back byte-identical to the input, untouched by the network.
    torch.manual_seed(0)
    model = MitigatorNet()
    frames = _flat_frames(20, 16, 16)
    result = mitigate_segment(frames, fps=10.0, profile=PROFILE, model=model)
    assert np.array_equal(result[0], frames[0])


def test_mitigate_segment_passes_through_frame_on_nan_output(monkeypatch):
    # An untrained or misbehaving model can output NaN/Inf. mitigate_segment
    # must never let that reach the returned frames -- fall back to the
    # original input for that frame instead.
    import mitigator.infer as infer_module

    def _nan_mitigate_frame(window, mask, histogram, model):
        return torch.full((1, 3, window.shape[-2], window.shape[-1]), float("nan"))

    monkeypatch.setattr(infer_module, "mitigate_frame", _nan_mitigate_frame)

    model = MitigatorNet()
    frames = _flat_frames(20, 16, 16)
    result = mitigate_segment(frames, fps=10.0, profile=PROFILE, model=model)
    for out_frame, in_frame in zip(result, frames):
        assert np.array_equal(out_frame, in_frame)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest mitigator/tests/test_infer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mitigator.infer'`

- [ ] **Step 4: Implement `mitigator/infer.py`**

```python
"""Evaluation-only inference wrapper: runs a Mitigator model over a
sequence of frames, sliding a 3-frame window and conditioning each frame on
PriorCalc's target histogram and risk mask. Not wired into any runtime
pipeline yet -- Verifier/Fallback/BufferManager integration is a separate,
later roadmap step (see docs/superpowers/specs/2026-08-03-mitigator-design.md
section 2). This exists so a trained checkpoint's output can be inspected
and measured (PSNR/SSIM vs ground truth) during and after training.
"""
import numpy as np
import torch

from detector.profiles import ThresholdProfile
from mitigator.arch import MitigatorNet, mitigate_frame
from prior.compute import compute_prior


def mitigate_segment(
    frames: list[np.ndarray],
    fps: float,
    profile: ThresholdProfile,
    model: MitigatorNet,
) -> list[np.ndarray]:
    priors = compute_prior(frames, fps=fps, profile=profile)
    model.eval()
    n = len(frames)
    output = [frame.copy() for frame in frames]

    with torch.no_grad():
        for prior in priors:
            if prior.target_histogram is None:
                continue  # no conditioning available -- pass through unchanged

            i = prior.frame_index
            prev_idx = max(0, i - 1)
            next_idx = min(n - 1, i + 1)
            window_np = np.concatenate([frames[prev_idx], frames[i], frames[next_idx]], axis=-1)

            window = torch.from_numpy(window_np.transpose(2, 0, 1)).float().unsqueeze(0)
            mask = torch.from_numpy(prior.mask[None, None, :, :].astype(np.float32))
            histogram = torch.from_numpy(prior.target_histogram).float().unsqueeze(0)

            restored = mitigate_frame(window, mask, histogram, model)
            if not torch.isfinite(restored).all():
                continue  # NaN/Inf from the model -- keep the original frame, never propagate garbage
            output[i] = restored.squeeze(0).permute(1, 2, 0).numpy()

    return output
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest mitigator/tests/test_infer.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add .gitignore mitigator/weights/.gitkeep mitigator/infer.py mitigator/tests/test_infer.py
git commit -m "feat(mitigator): add evaluation-only inference wrapper"
```

---

## Task 7: Package setup + full-suite verification

**Files:**
- Verify: `mitigator/__init__.py`, `mitigator/tests/__init__.py` (created in Task 2 — confirm present and empty)
- No new source files

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: nothing new — this task is verification only.

- [ ] **Step 1: Confirm package markers exist**

Run: `ls mitigator/__init__.py mitigator/tests/__init__.py`
Expected: both files exist (created in Task 2, Step 2)

- [ ] **Step 2: Run the full repository test suite**

Run: `pytest -v` from the repo root
Expected: all tests pass, with no GPU required (the new Mitigator tests all run on CPU with tiny synthetic tensors/frames). Note the total count — it will be the pre-Mitigator baseline (142, per the realistic-injection plan's final count) plus this plan's new tests.

- [ ] **Step 3: Confirm `torch` installs cleanly from `requirements.txt` alone**

Run: `pip install -r requirements.txt` (in a way that doesn't disturb the already-working environment — e.g. `pip install --dry-run -r requirements.txt` if available, or simply confirm `torch>=2.5` is satisfied by the already-installed `2.5.1+cu121`: `python -c "import torch; print(torch.__version__)"`)
Expected: no errors; torch version printed

- [ ] **Step 4: Manual verification (not an automated test) — smoke test on real data**

This step is a manual, one-time check run directly in the terminal, not part of the pytest suite. It confirms the pipeline works end-to-end against the real 170-sample `data/synthetic/` directory produced by DatasetSynth, and reports how long one real epoch takes on the actual GPU (RTX 3050 Laptop, 4GB VRAM) — informing, but not blocking, any future decision about full multi-epoch training.

```bash
python -c "
import time
from pathlib import Path
import torch
from detector.profiles import load_profile
from mitigator.arch import MitigatorNet
from training.mitigator_dataset import MitigatorDataset
from training.train_mitigator import train_one_epoch, evaluate
from torch.utils.data import DataLoader

profile = load_profile(Path('configs/profiles/itu.json'))
train_ds = MitigatorDataset(Path('data/synthetic'), profile, split='train')
val_ds = MitigatorDataset(Path('data/synthetic'), profile, split='val')
print(f'train examples: {len(train_ds)}, val examples: {len(val_ds)}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MitigatorNet()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

start = time.time()
train_loss = train_one_epoch(model, optimizer, train_loader, device=device)
elapsed = time.time() - start
val_metrics = evaluate(model, val_loader, device=device)
print(f'1 epoch: {elapsed:.1f}s, train_loss={train_loss:.4f}, val_loss={val_metrics[\"loss\"]:.4f}, val_psnr={val_metrics[\"psnr\"]:.2f}')
"
```

Expected: runs to completion without error, prints example counts, one epoch's elapsed time, and loss/PSNR values (any finite numbers — this step is about confirming the pipeline runs correctly and measuring real timing, not about hitting any particular quality bar). Record the timing in the SDD ledger for later reference when deciding whether/how to run full multi-epoch training.

- [ ] **Step 5: Final commit (if any files changed during verification)**

```bash
git status --short
```

If nothing changed, no commit needed — this task is verification-only. If `mitigator/weights/latest.pt` or `best.pt` were written by Step 4's manual smoke test, do NOT commit them (they are gitignored per Task 6) — they can be left on disk or deleted, at the implementer's discretion, since they are throwaway smoke-test artifacts, not a real trained model.
