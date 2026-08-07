# Self-Supervised Pipeline Implementation Plan (Plan B of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewire PriorCalc, the model's conditioning input, the dataset, and the training loop so the self-supervised loss built in Plan A can actually train a model.

**Architecture:** Additive first, then switch, then remove. PriorCalc gains `required_strength` while keeping `target_histogram`; the dataset gains `strength`, `dx`, `dy` alongside its existing fields; only then does the model swap its histogram embedding for a strength channel and the training loop swap its objective. The dead fields come out last. Every task ends with a green suite — a single sweeping change cannot.

**Tech Stack:** Python 3.12, PyTorch, NumPy, OpenCV (`cv2`), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-07-self-supervised-mitigation-design.md` (sections 4–7)

**Depends on:** Plan A, merged. `training.mitigator_losses.self_supervised_loss`, `training.mitigator_losses.warp_by_shift`, `training.soft_risk`, and the confidence guard in `detector.motion.estimate_global_shift` are all on `master`.

## Global Constraints

- Frames are RGB `float32` in `[0, 1]`, shape `(H, W, 3)` for NumPy and `(B, 3, H, W)` for torch.
- `detector/` stays NumPy-only. Torch lives in `training/` and `mitigator/`.
- Motion compensation is global shift only, from `detector.motion` — never dense optical flow.
- Every task must leave `python -m pytest` green. The suite is 313 passed / 1 deselected at the base commit.
- `pytest.ini` sets `pythonpath = .` and defaults to `-m "not slow"`. Run from the repo root.
- Any change to `prior/compute.py`'s output invalidates every `data/synthetic/*/prior_cache.npz`. The cache carries a schema version so a stale cache fails loudly instead of being silently misread.
- Evaluation clips are never trained on: `test_mp4/원본.mp4`, `test_mp4/Anyma*.mp4`, `test_mp4/Cera Khin*.mp4`, and everything in `data/raw/real_flicker_eval/`.
- Report contrast, not mean luminance, as the damage metric. The correction compresses toward a trailing mean, so mean luminance is near-invariant by construction and reporting it as a cost flatters the result.

---

### Task 1: PriorCalc produces a required strength

**Files:**
- Modify: `prior/compute.py`
- Modify: `training/mitigator_dataset.py` (cache read/write and `_build_example`)
- Test: `prior/tests/test_compute.py`, `training/tests/test_mitigator_dataset.py`

**Interfaces:**
- Consumes: `fallback.strength.required_strength(frames, profile) -> float` (on master).
- Produces: `FramePrior` gains `required_strength: float`. `MitigatorDataset.__getitem__` gains a `"strength"` key, a `(1,)` float32 tensor. `target_histogram` and `"histogram"` are untouched and still populated — Task 5 removes them.

- [ ] **Step 1: Write the failing tests**

Append to `prior/tests/test_compute.py`:

```python
def test_frame_prior_carries_the_segments_required_strength():
    # The model sees a three-frame window; how hard this segment needs to be
    # pushed is global context it cannot derive locally. Measured: the
    # previous model reached a median 14.6% of its own target, and a
    # histogram describing "what normal looks like" never told it how far to
    # move.
    import numpy as np
    from prior.compute import compute_prior

    rng = np.random.default_rng(0)
    base = rng.random((24, 24, 3)).astype(np.float32) * 0.2
    frames = []
    for i in range(40):
        frame = base.copy()
        if 15 <= i < 30 and i % 2 == 0:
            frame[:, :16] = 0.95
        frames.append(frame)

    priors = compute_prior(frames, fps=20.0, profile=PROFILE)

    flagged = [p for p in priors if p.required_strength > 0.0]
    assert flagged, "the flickering stretch should carry a nonzero strength"
    assert all(0.0 <= p.required_strength <= 1.0 for p in priors)


def test_frames_outside_a_risk_segment_have_zero_strength():
    # A frame the Detector did not flag needs no correction, and a nonzero
    # strength there would tell the model to move a frame that is already
    # compliant.
    import numpy as np
    from prior.compute import compute_prior

    rng = np.random.default_rng(1)
    frames = [rng.random((24, 24, 3)).astype(np.float32) * 0.1 for _ in range(30)]

    priors = compute_prior(frames, fps=20.0, profile=PROFILE)
    assert all(p.required_strength == 0.0 for p in priors)


def test_every_frame_in_one_segment_shares_that_segments_strength():
    # required_strength is a per-segment quantity: it is the max over the
    # segment's frame pairs, so handing different frames of one segment
    # different values would be incoherent.
    import numpy as np
    from prior.compute import compute_prior

    rng = np.random.default_rng(2)
    base = rng.random((24, 24, 3)).astype(np.float32) * 0.2
    frames = []
    for i in range(40):
        frame = base.copy()
        if 15 <= i < 30 and i % 2 == 0:
            frame[:, :16] = 0.95
        frames.append(frame)

    priors = compute_prior(frames, fps=20.0, profile=PROFILE)
    values = {round(p.required_strength, 6) for p in priors if p.required_strength > 0.0}
    assert len(values) == 1, f"expected one strength across the segment, got {values}"
```

If `prior/tests/test_compute.py` has no `PROFILE` at module level, add:

```python
from detector.profiles import ThresholdProfile

PROFILE = ThresholdProfile(
    name="test", max_flashes_per_second=3, max_area_ratio=0.25,
    general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
    red_saturation_ratio_threshold=0.80,
)
```

Append to `training/tests/test_mitigator_dataset.py`:

```python
def test_getitem_carries_the_required_strength(tmp_path):
    _write_fake_sample(tmp_path, "clipS", n_frames=20, h=8, w=8, fps=10.0,
                       segments=[{"start_frame": 12, "end_frame": 15}])
    dataset = MitigatorDataset(tmp_path, PROFILE, split=clip_split("clipS"))
    example = dataset[0]
    assert example["strength"].shape == (1,)
    assert 0.0 <= float(example["strength"]) <= 1.0
    assert example["strength_partner"].shape == (1,)


def test_a_stale_prior_cache_is_rejected_rather_than_misread(tmp_path):
    # The cache stores derived arrays with no self-description. A cache
    # written before required_strength existed would otherwise load with a
    # missing field and fail somewhere far from the cause.
    import numpy as np
    from training.mitigator_dataset import _load_prior_cache

    stale = tmp_path / "prior_cache.npz"
    np.savez_compressed(
        stale,
        has_target=np.array([True]),
        histograms=np.zeros((1, 64), dtype=np.float32),
        masks=np.ones((1, 4, 4), dtype=bool),
        frame_indices=np.array([0], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="prior cache"):
        _load_prior_cache(stale)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest prior/tests/test_compute.py training/tests/test_mitigator_dataset.py -v`
Expected: FAIL — `AttributeError: 'FramePrior' object has no attribute 'required_strength'` and `KeyError: 'strength'`.

- [ ] **Step 3: Add the field to `FramePrior` and populate it**

In `prior/compute.py`, add to the imports:

```python
from detector.segments import RiskSegment
from fallback.strength import required_strength
```

Add the field to the dataclass, after `mask`:

```python
    # How far this frame's risk segment has to be pushed, as a fraction of
    # its per-pixel luminance deltas -- 0.0 outside any segment.
    #
    # The model sees a three-frame window and cannot tell from it how severe
    # the surrounding segment is. The old conditioning was a histogram of
    # "what normal illumination looks like", which describes a destination
    # but not a distance; the trained model reached a median 14.6% of it.
    # This is the distance, derived from the same quantity the Detector
    # thresholds on. Apple's published mitigation scales its strength the
    # same way, from a measured risk score rather than a fixed constant.
    required_strength: float = 0.0
```

At the end of `compute_prior`, before `return results`, replace the return with:

```python
    _assign_segment_strengths(frames, results, segments, profile)
    return results
```

and change the detection call to keep the segments:

```python
    scores, segments, masks = run_detection_with_masks(frames, fps=fps, profile=profile)
```

Then add the helper above `compute_prior`:

```python
def _assign_segment_strengths(
    frames: list[np.ndarray],
    priors: list[FramePrior],
    segments: list[RiskSegment],
    profile: ThresholdProfile,
) -> None:
    """Give every frame of a risk segment that segment's required strength.

    The strength is a per-segment quantity by construction --
    `fallback.strength.required_strength` takes the maximum over the
    segment's frame pairs, because a segment is flagged if any pair in it
    trips the area limit. Handing different frames of one segment different
    values would be incoherent, so they all get the segment's value.
    """
    for segment in segments:
        window = frames[segment.start_frame : segment.end_frame + 1]
        strength = required_strength(window, profile)
        for index in range(segment.start_frame, segment.end_frame + 1):
            priors[index].required_strength = strength
```

- [ ] **Step 4: Version the cache and carry the field through it**

In `training/mitigator_dataset.py`, add near the other module constants:

```python
# Bumped whenever prior_cache.npz's contents change meaning. A cache
# written by an older version is rejected rather than loaded with missing
# or misinterpreted fields -- the failure would otherwise surface as bad
# training rather than an error.
_PRIOR_CACHE_VERSION = 2
```

In `_save_prior_cache`, add to the `np.savez_compressed` call:

```python
        version=np.array([_PRIOR_CACHE_VERSION], dtype=np.int64),
        required_strengths=np.array([p.required_strength for p in priors], dtype=np.float32),
```

Replace `_load_prior_cache`'s body with:

```python
def _load_prior_cache(cache_path: Path) -> list[FramePrior]:
    with np.load(cache_path) as data:
        version = int(data["version"][0]) if "version" in data else 1
        if version != _PRIOR_CACHE_VERSION:
            raise ValueError(
                f"prior cache {cache_path} is version {version}, expected "
                f"{_PRIOR_CACHE_VERSION}. Delete every data/synthetic/*/prior_cache.npz "
                f"and let training regenerate them."
            )
        frame_indices = data["frame_indices"]
        histograms = data["histograms"]
        has_target = data["has_target"]
        masks = data["masks"]
        strengths = data["required_strengths"]
    priors = []
    for i in range(len(frame_indices)):
        target = histograms[i] if has_target[i] else None
        priors.append(FramePrior(
            frame_index=int(frame_indices[i]),
            target_histogram=target,
            mask=masks[i],
            required_strength=float(strengths[i]),
        ))
    return priors
```

In `_build_example`, add to the returned dict:

```python
            "strength": torch.tensor([prior.required_strength], dtype=torch.float32),
```

In `__getitem__`, add alongside the other partner fields:

```python
        example["strength_partner"] = partner["strength"]
```

- [ ] **Step 5: Delete every stale cache, then run the full suite**

```bash
find data/synthetic -name prior_cache.npz -delete
python -m pytest -q
```

Expected: PASS, 318 tests (313 + 5 new).

The cache deletion is not optional. Any surviving v1 cache now raises, which is the intended behaviour, but leaving them in place means the next training run spends its first minutes failing instead of recomputing.

- [ ] **Step 6: Commit**

```bash
git add prior/compute.py training/mitigator_dataset.py prior/tests/test_compute.py training/tests/test_mitigator_dataset.py
git commit -m "feat(prior): carry each segment's required correction strength"
```

---

### Task 2: The dataset carries the motion estimate

The temporal term needs the shift between the two frames it compares. That is a `cv2` call, so it belongs in the dataset — computed once and cached — not in the training loop.

**Files:**
- Modify: `training/mitigator_dataset.py`
- Test: `training/tests/test_mitigator_dataset.py`

**Interfaces:**
- Consumes: `detector.motion.estimate_global_shift(prev_gray, curr_gray, min_response=0.30) -> tuple[float, float]`, and `detector.luminance.relative_luminance`.
- Produces: `__getitem__` gains `"shift"`, a `(2,)` float32 tensor `[dx, dy]` from the centre frame to its partner, and `"shift_trusted"`, a `(1,)` float32 tensor that is 1.0 when phase correlation was confident and 0.0 when its estimate was rejected.

- [ ] **Step 1: Write the failing tests**

Append to `training/tests/test_mitigator_dataset.py`:

```python
def test_getitem_carries_the_shift_to_its_partner(tmp_path):
    _write_fake_sample(tmp_path, "clipM", n_frames=20, h=16, w=16, fps=10.0,
                       segments=[{"start_frame": 12, "end_frame": 15}])
    dataset = MitigatorDataset(tmp_path, PROFILE, split=clip_split("clipM"))
    example = dataset[0]
    assert example["shift"].shape == (2,)
    assert example["shift_trusted"].shape == (1,)
    assert float(example["shift_trusted"]) in (0.0, 1.0)


def test_a_self_paired_frame_reports_no_shift(tmp_path):
    # At a segment's last frame the partner is the frame itself, so there is
    # no motion between them by construction. A nonzero shift there would
    # warp a frame against itself and charge the model for the difference.
    _write_fake_sample(tmp_path, "clipN", n_frames=20, h=16, w=16, fps=10.0,
                       segments=[{"start_frame": 12, "end_frame": 15}])
    dataset = MitigatorDataset(tmp_path, PROFILE, split=clip_split("clipN"))
    frame_indices = [idx for _, idx in dataset._index]
    example = dataset[frame_indices.index(max(frame_indices))]
    assert float(example["shift"][0]) == 0.0
    assert float(example["shift"][1]) == 0.0


def test_an_untrusted_shift_is_flagged_rather_than_silently_zeroed(tmp_path, monkeypatch):
    # estimate_global_shift returns (0.0, 0.0) both when the frames really
    # did not move and when it refused to guess. The loss cannot tell those
    # apart, and on a rejected pan it would charge real camera motion as
    # flicker -- with none of the Detector's three smoothing stages to
    # absorb it. Measured: 33.6% of pairs rejected on one real clip.
    import training.mitigator_dataset as dataset_module

    monkeypatch.setattr(dataset_module, "estimate_global_shift", lambda a, b: (0.0, 0.0))
    monkeypatch.setattr(dataset_module, "_shift_is_trusted", lambda a, b: False)

    _write_fake_sample(tmp_path, "clipU", n_frames=20, h=16, w=16, fps=10.0,
                       segments=[{"start_frame": 12, "end_frame": 15}])
    dataset = MitigatorDataset(tmp_path, PROFILE, split=clip_split("clipU"))
    assert float(dataset[0]["shift_trusted"]) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest training/tests/test_mitigator_dataset.py -v`
Expected: FAIL — `KeyError: 'shift'`.

- [ ] **Step 3: Implement**

Add to `training/mitigator_dataset.py`'s imports:

```python
import cv2

from detector.luminance import relative_luminance
from detector.motion import estimate_global_shift
```

Add above `MitigatorDataset`:

```python
# Same default the Detector uses. Duplicated as a call rather than a
# constant so the two can never drift: `estimate_global_shift` already
# applies it, and this only asks whether it fired.
def _shift_is_trusted(prev_luminance: np.ndarray, curr_luminance: np.ndarray) -> bool:
    """Whether phase correlation actually locked onto structure.

    `estimate_global_shift` returns (0.0, 0.0) both when the frames genuinely
    did not move and when it refused to guess, and the loss cannot tell those
    apart. On a rejected pan it would warp by zero and then charge the real
    camera motion as flicker -- and unlike the Detector, which smooths a bad
    warp through thresholding, area-averaging and a one-second windowed
    maximum, the temporal term is a raw per-pixel L1 with no such stage.
    """
    _shift, response = cv2.phaseCorrelate(
        prev_luminance.astype(np.float32), curr_luminance.astype(np.float32)
    )
    return response >= 0.30
```

In `__getitem__`, after the partner is built, replace the tail with:

```python
        example["window_partner"] = partner["window"]
        example["mask_partner"] = partner["mask"]
        example["histogram_partner"] = partner["histogram"]
        example["clean_partner"] = partner["clean"]
        example["strength_partner"] = partner["strength"]

        if partner_index == frame_index:
            # Self-paired at a segment's last frame: no motion by
            # construction, and trusted precisely because it is not a guess.
            example["shift"] = torch.zeros(2, dtype=torch.float32)
            example["shift_trusted"] = torch.ones(1, dtype=torch.float32)
        else:
            centre = relative_luminance(_read_frame(sample_dir / "degraded", frame_index))
            follower = relative_luminance(_read_frame(sample_dir / "degraded", partner_index))
            dx, dy = estimate_global_shift(centre, follower)
            example["shift"] = torch.tensor([dx, dy], dtype=torch.float32)
            example["shift_trusted"] = torch.tensor(
                [1.0 if _shift_is_trusted(centre, follower) else 0.0], dtype=torch.float32
            )
        return example
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS, 321 tests (318 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add training/mitigator_dataset.py training/tests/test_mitigator_dataset.py
git commit -m "feat(training): carry the motion estimate and its trust flag per example"
```

---

### Task 3: The model is conditioned on strength, not a histogram

**Files:**
- Modify: `mitigator/arch.py`
- Modify: `mitigator/infer.py`
- Modify: `mitigator/cli.py` if it constructs conditioning
- Modify: `training/train_mitigator.py` (call sites only)
- Test: `mitigator/tests/test_arch.py`, and every test that constructs `MitigatorNet` or calls `mitigate_frame`

**Interfaces:**
- Consumes: `"strength"` from the dataset (Task 1).
- Produces: `MitigatorNet.forward(window, mask, strength)` and `mitigate_frame(window, mask, strength, model)`. `strength` is `(B, 1)`. `in_channels` becomes `3 * 3 + 1 + 1 = 11`.

- [ ] **Step 1: Write the failing tests**

Append to `mitigator/tests/test_arch.py`:

```python
def test_forward_accepts_a_strength_scalar_instead_of_a_histogram():
    from mitigator.arch import MitigatorNet

    model = MitigatorNet()
    window = torch.rand(2, 9, 32, 32)
    mask = torch.ones(2, 1, 32, 32)
    strength = torch.tensor([[0.3], [0.9]])
    assert model(window, mask, strength).shape == (2, 3, 32, 32)


def test_strength_changes_the_output():
    # Conditioning the model on a value it ignores would be worse than not
    # conditioning it at all -- it would look informed and behave otherwise.
    from mitigator.arch import MitigatorNet

    torch.manual_seed(0)
    model = MitigatorNet()
    window = torch.rand(1, 9, 32, 32)
    mask = torch.ones(1, 1, 32, 32)

    low = model(window, mask, torch.tensor([[0.1]]))
    high = model(window, mask, torch.tensor([[0.9]]))
    assert not torch.allclose(low, high, atol=1e-5)


def test_mitigate_frame_takes_strength():
    from mitigator.arch import MitigatorNet, mitigate_frame

    model = MitigatorNet()
    window = torch.rand(1, 9, 16, 16)
    mask = torch.zeros(1, 1, 16, 16)
    result = mitigate_frame(window, mask, torch.tensor([[0.5]]), model)
    # mask is all zeros, so the hard blend must return the centre frame
    # untouched regardless of what the network produced.
    assert torch.allclose(result, window[:, 3:6], atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest mitigator/tests/test_arch.py -v`
Expected: FAIL — the histogram embedding rejects a `(2, 1)` input.

- [ ] **Step 3: Implement**

In `mitigator/arch.py`, delete `_HIST_BINS` and `_HIST_EMBED_CHANNELS`, delete `self.hist_embed`, and replace the `__init__` line and `forward`:

```python
        # 3 RGB frames + risk mask + required strength.
        #
        # The strength replaces a 64-bin illumination histogram. That
        # histogram described what normal illumination looks like, derived
        # from the same frame's non-masked pixels -- sound on synthetic data,
        # where shapes are pasted onto ordinary video, but wrong on real
        # concert footage, where inside the mask is stage lighting and
        # outside is a dark crowd. Two different objects. It asked the model
        # to make the lights as dark as the audience, and the model followed
        # it to a median 14.6%.
        in_channels = 3 * 3 + 1 + 1
```

```python
    def forward(
        self,
        window: torch.Tensor,
        mask: torch.Tensor,
        strength: torch.Tensor,
    ) -> torch.Tensor:
        b, _, h, w = window.shape
        strength_spatial = strength.view(b, 1, 1, 1).expand(-1, -1, h, w)
        x = torch.cat([window, mask, strength_spatial], dim=1)

        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        x3 = self.bottleneck(x3)
        x = self.up3(x3, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        return self.out_conv(x)
```

Rename the third parameter of `mitigate_frame` from `histogram` to `strength` and pass it through.

- [ ] **Step 4: Update every call site and test**

`mitigator/infer.py` and `training/train_mitigator.py` call `mitigate_frame(window, mask, histogram, model)`. Change the third argument to the strength tensor. In `infer.py` the prior now supplies `required_strength`; build `torch.tensor([[prior.required_strength]])` on the model's device.

Search for remaining references and fix each:

```bash
grep -rn "histogram" --include="*.py" mitigator/ training/ | grep -v mitigator_dataset
```

`training/train_mitigator.py` currently passes `batch["histogram"]`; change to `batch["strength"]`.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS, 324 tests (321 + 3 new).

`mitigator/weights/best.pt` can no longer be loaded — `in_channels` changed, so the first convolution's weight shape differs. That is expected; the checkpoint is kept for comparison, not for resuming.

- [ ] **Step 6: Commit**

```bash
git add mitigator/ training/train_mitigator.py
git commit -m "feat(mitigator): condition on the required strength instead of a target histogram"
```

---

### Task 4: The training loop uses the self-supervised loss

**Files:**
- Modify: `training/train_mitigator.py`
- Test: `training/tests/test_train_mitigator.py`

**Interfaces:**
- Consumes: `training.mitigator_losses.self_supervised_loss(out_a, out_b, in_a, in_b, dx, dy, profile, lambda_temporal, lambda_risk, tau, tau_area) -> dict` with keys `loss`, `fidelity`, `temporal`, `risk`, `soft_area`.
- Produces: `--lambda-temporal`, `--lambda-risk`, `--tau`, `--tau-area` CLI arguments; the epoch log gains `val_fidelity`, `val_temporal`, `val_risk`, `val_soft_area`.

- [ ] **Step 1: Write the failing tests**

Append to `training/tests/test_train_mitigator.py`:

```python
class _SelfSupervisedBatchDataset(torch.utils.data.Dataset):
    """A flickering pair with everything the self-supervised loss needs."""

    def __init__(self, h=16, w=16, n=4):
        torch.manual_seed(0)
        dark = torch.rand(3, h, w) * 0.2
        bright = (dark + 0.7).clamp(0, 1)
        self.window = torch.cat([dark, dark, bright], dim=0)
        self.window_partner = torch.cat([dark, bright, bright], dim=0)
        self.mask = torch.ones(1, h, w)
        self.strength = torch.tensor([0.6])
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "window": self.window, "mask": self.mask, "strength": self.strength,
            "window_partner": self.window_partner, "mask_partner": self.mask,
            "strength_partner": self.strength,
            "shift": torch.zeros(2), "shift_trusted": torch.ones(1),
        }


def test_train_one_epoch_runs_on_the_self_supervised_loss():
    from training.train_mitigator import train_one_epoch

    torch.manual_seed(0)
    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(_SelfSupervisedBatchDataset(), batch_size=2)
    profile = load_profile(Path("configs/profiles/itu.json"))

    loss = train_one_epoch(model, optimizer, loader, device="cpu", profile=profile)
    assert loss == loss  # NaN check
    assert loss > 0.0


def test_evaluate_reports_every_loss_term():
    # The training log is how the fidelity/risk trade-off is watched. A
    # combined number alone hides which way it is going, and lambda_temporal
    # is the over-correction knob -- measured, not assumed.
    from training.train_mitigator import evaluate

    model = MitigatorNet()
    loader = torch.utils.data.DataLoader(_SelfSupervisedBatchDataset(), batch_size=1)
    profile = load_profile(Path("configs/profiles/itu.json"))

    metrics = evaluate(model, loader, device="cpu", profile=profile)
    for key in ("loss", "fidelity", "temporal", "risk", "soft_area"):
        assert key in metrics


def test_an_untrusted_shift_drops_the_temporal_and_risk_terms():
    # A rejected phase-correlation estimate is indistinguishable from "no
    # motion" once it is a zero shift, so on a rejected pan both terms would
    # charge real camera movement as flicker. The example carries a trust
    # flag; the loop must honour it.
    from training.train_mitigator import compute_batch_losses

    model = MitigatorNet()
    profile = load_profile(Path("configs/profiles/itu.json"))
    batch = torch.utils.data.default_collate([_SelfSupervisedBatchDataset()[0]])
    batch["shift_trusted"] = torch.zeros(1, 1)

    terms = compute_batch_losses(batch, model, "cpu", profile)
    assert terms["temporal"].item() == pytest.approx(0.0, abs=1e-6)
    assert terms["risk"].item() == pytest.approx(0.0, abs=1e-6)
    assert terms["fidelity"].item() > 0.0 or terms["loss"].item() >= 0.0
```

Add `from pathlib import Path` and `from detector.profiles import load_profile` to the test file's imports if absent.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest training/tests/test_train_mitigator.py -v`
Expected: FAIL — `train_one_epoch` has no `profile` parameter and `compute_batch_losses` does not exist.

- [ ] **Step 3: Implement**

In `training/train_mitigator.py`, replace `compute_losses` with:

```python
def compute_batch_losses(
    batch: dict,
    model: MitigatorNet,
    device: str,
    profile: ThresholdProfile,
    lambda_temporal: float = 1.0,
    lambda_risk: float = 1.0,
    tau: float = 0.02,
    tau_area: float = 0.05,
) -> dict:
    """Restore the centre frame and its partner, and score them without a
    clean reference.

    Both go through the model in one concatenated forward pass -- same
    arithmetic, one kernel launch sequence.

    An example whose `shift_trusted` is 0 has both motion-dependent terms
    zeroed. `estimate_global_shift` returns (0, 0) both when the frames
    genuinely did not move and when it refused to guess, and on a rejected
    pan the temporal and risk terms would charge real camera motion as
    flicker. The Detector absorbs a bad warp through thresholding,
    area-averaging and a one-second windowed maximum; this loss has none of
    those stages. Measured: 33.6% of frame pairs rejected on one real clip.
    """
    n = batch["window"].shape[0]
    window = torch.cat([batch["window"], batch["window_partner"]], dim=0).to(device)
    mask = torch.cat([batch["mask"], batch["mask_partner"]], dim=0).to(device)
    strength = torch.cat([batch["strength"], batch["strength_partner"]], dim=0).to(device)

    restored = mitigate_frame(window, mask, strength, model)
    out_a, out_b = restored[:n], restored[n:]
    in_a = window[:n, 3:6]
    in_b = window[n:, 3:6]

    shift = batch["shift"].to(device)
    trusted = batch["shift_trusted"].to(device).view(-1)

    terms = self_supervised_loss(
        out_a, out_b, in_a, in_b, shift[:, 0], shift[:, 1], profile,
        lambda_temporal=lambda_temporal, lambda_risk=lambda_risk,
        tau=tau, tau_area=tau_area,
    )

    gate = trusted.mean()
    return {
        "loss": terms["fidelity"] + gate * (lambda_temporal * terms["temporal"]
                                            + lambda_risk * terms["risk"]),
        "fidelity": terms["fidelity"],
        "temporal": gate * terms["temporal"],
        "risk": gate * terms["risk"],
        "soft_area": terms["soft_area"],
    }
```

Update `train_one_epoch` and `evaluate` to take `profile` plus the four weights and call `compute_batch_losses`. `evaluate` returns the mean of every key.

Add the argparse arguments:

```python
    parser.add_argument("--lambda-temporal", type=float, default=1.0,
                        help="Weight on temporal consistency. Measured to be the term that "
                             "actually moves the output (35-43%% of gradient at settling), and "
                             "therefore the over-correction knob -- raise lambda_risk to fix "
                             "under-correction and nothing happens.")
    parser.add_argument("--lambda-risk", type=float, default=1.0,
                        help="Weight on the threshold penalty. Load-bearing (at 0 the flicker "
                             "stays flagged at its input level) but a near-threshold finisher, "
                             "not a driver: at tau=0.02 strongly-flagged pixels sit deep in "
                             "sigmoid saturation.")
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--tau-area", type=float, default=0.05)
```

Add every term to the JSON log line and the printed line.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS, 327 tests (324 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add training/train_mitigator.py training/tests/test_train_mitigator.py
git commit -m "feat(training): train on the self-supervised loss"
```

---

### Task 5: Remove what the new path does not use

**Files:**
- Modify: `prior/compute.py`, `training/mitigator_dataset.py`, `training/mitigator_losses.py`
- Delete: `prior/histogram.py`, `prior/tests/test_histogram.py`
- Test: the existing suites

**Interfaces:**
- Consumes: nothing new.
- Produces: `FramePrior` loses `target_histogram`. `MitigatorDataset.__getitem__` loses `"histogram"`, `"histogram_partner"`, `"clean"`, `"clean_partner"`. `temporal_consistency_loss` is removed.

- [ ] **Step 1: Confirm nothing still reads them**

```bash
grep -rn "target_histogram\|\"histogram\"\|\"clean\"\|temporal_consistency_loss\|compute_illumination_histogram" --include="*.py" .
```

Every hit must be inside the files this task deletes or edits. If a hit is anywhere else, stop and report — something still depends on the old path and removing it here would be a silent break rather than a cleanup.

- [ ] **Step 2: Remove the histogram from PriorCalc**

Delete the `target_histogram` field, the `n_bins` parameter, the `window`/`last_valid_target` machinery, and the `compute_illumination_histogram` import from `prior/compute.py`. `_MIN_REFERENCE_PIXEL_RATIO` and `DEFAULT_N_BINS` go with them. Keep the mask persistence — it is unrelated and still justified.

Delete `prior/histogram.py` and `prior/tests/test_histogram.py`.

Remove `has_target`, `histograms` and the `n_bins` handling from `_save_prior_cache` / `_load_prior_cache`, and bump `_PRIOR_CACHE_VERSION` to `3`.

- [ ] **Step 3: Remove the clean-frame reads from the dataset**

In `_build_example`, drop the `clean_center` read and the `"histogram"` and `"clean"` keys. In `__getitem__`, drop `"histogram_partner"` and `"clean_partner"`. The `clean/` directories stay on disk — the spec keeps them until the loss design is validated in training.

Update the dataset's `__init__` filter: it currently requires `prior.target_histogram is not None`. Replace that condition with membership in a risk segment alone, and update the module docstring, which describes the old histogram-conditioned selection.

- [ ] **Step 4: Remove the superseded loss**

Delete `temporal_consistency_loss` from `training/mitigator_losses.py` and its tests. Design section 3.2 rejects its formulation: it compares the prediction's delta against a *target* delta, which makes flicker free because the flicker is in the target too. Leaving both live means two functions in one file each arguing the other is wrong.

- [ ] **Step 5: Delete every stale cache and run the full suite**

```bash
find data/synthetic -name prior_cache.npz -delete
python -m pytest -q
```

Expected: PASS. The count drops as the histogram and old-loss tests go; report the number you get rather than matching a predicted one.

- [ ] **Step 6: Commit**

```bash
git add -A prior/ training/
git commit -m "refactor: drop the target histogram, the clean-frame label, and the superseded temporal loss"
```

---

### Task 6: Evaluation reports the cost that matters

`scripts/evaluate_mitigation.py` reports mean luminance, which this correction leaves near-invariant by construction. Contrast is the honest figure, and it is what the neural path has to beat.

**Files:**
- Modify: `scripts/evaluate_mitigation.py`
- Test: `scripts/tests/test_evaluate_mitigation.py` (create the package if absent)

**Interfaces:**
- Consumes: `detector.luminance.relative_luminance`, `detector.pipeline.run_detection`.
- Produces: `contrast_stats(frames) -> dict` with keys `mean_contrast` and `p95_contrast`; the printed report gains a contrast block and a comparison against the Tier 0 baseline.

- [ ] **Step 1: Write the failing test**

Create `scripts/tests/__init__.py` (empty) and `scripts/tests/test_evaluate_mitigation.py`:

```python
import numpy as np

from scripts.evaluate_mitigation import contrast_stats


def test_contrast_falls_when_a_frame_is_compressed_toward_its_mean():
    rng = np.random.default_rng(0)
    frames = [rng.random((32, 32, 3)).astype(np.float32) for _ in range(5)]
    flattened = [f * 0.2 + f.mean() * 0.8 for f in frames]

    assert contrast_stats(flattened)["mean_contrast"] < contrast_stats(frames)["mean_contrast"]


def test_mean_luminance_can_stay_flat_while_contrast_collapses():
    # This is why contrast is the reported cost and mean luminance is not.
    # Compressing toward the mean leaves the mean almost untouched -- on a
    # real clip it rose 1.87% while contrast fell 52.8%.
    from detector.luminance import relative_luminance
    from scripts.evaluate_mitigation import luminance_stats

    rng = np.random.default_rng(1)
    frames = [rng.random((32, 32, 3)).astype(np.float32) for _ in range(5)]
    flattened = [f * 0.1 + f.mean() * 0.9 for f in frames]

    before_mean = luminance_stats(frames)["mean_luminance"]
    after_mean = luminance_stats(flattened)["mean_luminance"]
    assert abs(after_mean - before_mean) / before_mean < 0.10

    before_contrast = contrast_stats(frames)["mean_contrast"]
    after_contrast = contrast_stats(flattened)["mean_contrast"]
    assert after_contrast < before_contrast * 0.5
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest scripts/tests/test_evaluate_mitigation.py -v`
Expected: FAIL — `ImportError: cannot import name 'contrast_stats'`.

- [ ] **Step 3: Implement**

Add to `scripts/evaluate_mitigation.py`:

```python
def contrast_stats(frames: list[np.ndarray]) -> dict:
    """Within-frame luminance spread, averaged over the clip.

    This is the damage figure to report, not mean luminance. The correction
    compresses each frame toward its own trailing mean, so the mean is
    near-invariant by construction -- on one real clip it rose 1.87% while
    contrast fell 52.8%. Reporting the mean as a cost makes a heavy
    correction look gentle.
    """
    spreads = np.array([float(relative_luminance(frame).std()) for frame in frames])
    return {
        "mean_contrast": float(spreads.mean()) if spreads.size else 0.0,
        "p95_contrast": float(np.percentile(spreads, 95)) if spreads.size else 0.0,
    }
```

In `_print_report`, add a contrast block after the luminance block, and a line comparing the measured contrast change against the Tier 0 baseline:

```python
    print("\n=== Within-frame contrast (the honest damage figure) ===")
    print(f"{'metric':<28}{'before':>10}{'after':>10}{'change':>10}")
    for label, key in [("mean contrast", "mean_contrast"), ("p95 contrast", "p95_contrast")]:
        print(f"{label:<28}{contrast_before[key]:>10.4f}{contrast_after[key]:>10.4f}"
              f"{change(contrast_before[key], contrast_after[key]):>10}")
    print("  Tier 0 baseline on real clips: -52.8% (Anyma), -48.4% (Cera Khin).")
    print("  A neural correction that passes but costs more than this loses to a")
    print("  classical method needing no GPU, no training data, and under 1 ms/frame.")
```

Compute `contrast_before` / `contrast_after` in `main` alongside the luminance stats and pass them to `_print_report`.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS, with 2 more tests than Task 5 left it at.

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate_mitigation.py scripts/tests/
git commit -m "feat(scripts): report within-frame contrast as the damage metric"
```

---

## Self-Review

**Spec coverage (sections 4–7):**

| Spec requirement | Task |
|---|---|
| 4 — target histogram removed | Task 5 |
| 4 — `FramePrior` carries mask + required_strength | Task 1 |
| 4 — strength is per-segment, 0 outside | Task 1 |
| 4 — mask persistence kept | Task 5 (explicitly preserved) |
| 4 — stale `prior_cache.npz` must be deleted | Tasks 1 and 5, with a version guard |
| 5 — `in_channels = 3*3 + 1 + 1` | Task 3 |
| 5 — histogram embedding removed | Task 3 |
| 5 — `best.pt` no longer loadable | Task 3, stated |
| 6.1 — phase 1 uses degraded frames only, `clean` untouched on disk | Task 5 |
| 7 — no PSNR; contrast is the cost | Task 6 |
| 7 — Tier 0 baseline −52.8% / −48.4% is the bar | Task 6 |
| 8 — confidence guard's `(0,0)` ambiguity | Tasks 2 and 4 |

Section 6.2 (the evaluation clips are never trained on) is a Global Constraint rather than a task: no task in this plan reads from `test_mp4/` or `data/raw/real_flicker_eval/`.

**Type consistency:** `"strength"` is `(1,)` in the dataset and `(B, 1)` after collation, which is what `MitigatorNet.forward` reshapes with `strength.view(b, 1, 1, 1)`. `"shift"` is `(2,)` per example, `(B, 2)` collated, split into `shift[:, 0]` and `shift[:, 1]` for `self_supervised_loss`'s `dx`/`dy`. `compute_batch_losses` is the name used in both Task 4's implementation and its tests.

**Deliberate deviation from the spec:** section 4 says PriorCalc produces "mask + strength", implying the histogram goes in the same change. This plan keeps it through Tasks 1–4 and removes it in Task 5, because removing it in Task 1 breaks the model, the dataset and the training loop simultaneously and no intermediate task could leave the suite green. The end state matches the spec.

**Not covered here, deliberately:** the training runs themselves (spec section 8 steps 6–7). Those are operations on VESSL, not implementation, and they depend on this plan landing first.
