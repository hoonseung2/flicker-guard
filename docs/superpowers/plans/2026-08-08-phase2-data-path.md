# Phase 2 Data Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make real flickering footage trainable — ingest 14 real clips into the layout `MitigatorDataset` already reads, exclude the Detector's warm-up frames, put train/dev membership under explicit control, and add `--init-from` and `--snapshot-every` to the trainer.

**Architecture:** A new one-way ingestion script (mp4 → sample directory) reusing `scripts.screen_clean_clips.read_frames_lenient` for decode and downscale. `MitigatorDataset` gains two behaviours behind optional parameters so the synthetic path keeps working byte-for-byte: risky frames fall back to the priors' own `required_strength` when `meta.json` has no `segments`, and an optional explicit split map replaces the `clip_id` hash. The trainer gains two flags that do not change any existing default.

**Tech Stack:** Python 3.10, PyTorch 2.11, OpenCV, NumPy, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-08-phase2-data-path-design.md`. Read §4–§8 before starting.
- Every text-file read and write passes `encoding="utf-8"` explicitly. Windows defaults to cp949 here and the corpus contains emoji and Korean; this has already caused a real decode failure.
- No existing default may change. `MitigatorDataset(data_dir, profile, split)` with three arguments, and `train_mitigator` without the new flags, must behave exactly as they do today — phase 1 has to stay reproducible.
- The repo has no configured git identity. Commit with:
  `git -c user.name="hoonseung2" -c user.email="dltmdgns0323@gmail.com" commit -m "..."`
- Run tests with `CUDA_VISIBLE_DEVICES="" python -m pytest <path> -q` unless the test needs a GPU. A training run may be using the GPU.
- Baseline before starting: 198 tests pass in `training/tests/` and `scripts/tests/`.
- Never read `data/raw/real_flicker_eval/` or `test_mp4/`. Those six clips are the held-out evaluation set.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/ingest_real_clips.py` (create) | mp4 directory → per-clip sample directories. Slug/id derivation, frame dump, `meta.json`. |
| `scripts/tests/test_ingest_real_clips.py` (create) | Tests for the above. |
| `configs/splits/phase2.json` (create) | Explicit `clip_id` → `"train" \| "val"` map. Data, not code, so the synthetic hash path is untouched. |
| `training/mitigator_dataset.py` (modify) | Risky-frame fallback, warm-up exclusion, optional explicit split map. |
| `training/tests/test_mitigator_dataset.py` (modify) | Tests for the above. |
| `training/train_mitigator.py` (modify) | `--init-from`, `--snapshot-every`, `--split-file`. |
| `training/tests/test_train_mitigator.py` (modify) | Tests for the above. |

### Two decisions this plan makes that the spec left open

**1. Risky frames without `meta["segments"]`.** `MitigatorDataset.__init__` currently builds `risky_indices` from `meta["segments"]`. Ingested real clips have no segments — computing them in the ingest script would duplicate the detection the dataset already runs through `compute_prior`.

Instead: when `segments` is absent, derive risky frames from `prior.required_strength > 0.0`. This is the same information. `prior/compute.py:_assign_segment_strengths` sets a non-zero strength on exactly the frames of a detected `RiskSegment`, and `mitigator/infer.py:mitigate_segment` already treats `required_strength > 0.0` as its "is this frame in a risk segment" gate. When `segments` IS present the existing path is used unchanged, so synthetic samples are unaffected.

**2. The split is still named `"val"`, not `"dev"`.** The spec calls the held-out three clips the dev set. Renaming the split value would touch `train_mitigator.main`, `evaluate`, and several tests for no behavioural gain. The code keeps `"train"`/`"val"`; for phase 2 the `"val"` split *is* the dev set.

---

## Task 1: `clip_id` derivation

**Files:**
- Create: `scripts/ingest_real_clips.py`
- Test: `scripts/tests/test_ingest_real_clips.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `clip_id_for(filename: str) -> str`. Later tasks and `configs/splits/phase2.json` use its output as the sample directory name and split key.

- [ ] **Step 1: Write the failing test**

```python
# scripts/tests/test_ingest_real_clips.py
from scripts.ingest_real_clips import clip_id_for


def test_clip_id_is_ascii_and_deterministic():
    name = "FE!N in London was epic \U0001f92f\U0001f4a5 #travisscott.mp4"
    first = clip_id_for(name)
    assert first == clip_id_for(name)
    assert first.isascii()
    assert first.encode("ascii")  # no surrogates


def test_clip_id_distinguishes_names_that_slug_identically():
    # Both collapse to the same slug -- only the hash keeps them apart.
    # A collision here would silently merge two clips into one sample dir.
    a = clip_id_for("녹음 2026-08-08 000241.mp4")
    b = clip_id_for("@@@@@ 2026-08-08 000241.mp4")
    assert a.rsplit("-", 1)[0] == b.rsplit("-", 1)[0]
    assert a != b


def test_clip_id_keeps_readable_stem_for_plain_names():
    assert clip_id_for("11832-233049403_medium.mp4").startswith("11832-233049403_medium-")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest scripts/tests/test_ingest_real_clips.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.ingest_real_clips'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/ingest_real_clips.py
"""Ingest already-flickering real clips into the sample layout
MitigatorDataset reads. No injection: these clips arrive degraded.

Phase 1 synthesised its samples by injecting flicker into clean sources.
Phase 2 must not -- so this is a separate one-way script rather than a
flag on training/cli.py, whose --profiles-dir / --samples-per-combo /
--injection-mode arguments would all become meaningless with injection
turned off.
"""
import hashlib
import re
from pathlib import Path

_NON_SLUG = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_SLUG = 60


def clip_id_for(filename: str) -> str:
    """Stable ASCII directory name for a source clip.

    The corpus contains emoji, '#', '!', Korean and doubled spaces.
    Emoji in a directory name is risky on Windows, and clip_id is the key
    the train/dev split is written against, so it has to be both ASCII and
    stable. Folding everything outside [A-Za-z0-9._-] to '_' can map two
    different filenames onto one slug, so a hash of the *full original
    filename* is appended to keep them distinct.
    """
    slug = _NON_SLUG.sub("_", Path(filename).stem).strip("._")[:_MAX_SLUG] or "clip"
    digest = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest scripts/tests/test_ingest_real_clips.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/ingest_real_clips.py scripts/tests/test_ingest_real_clips.py
git -c user.name="hoonseung2" -c user.email="dltmdgns0323@gmail.com" commit -m "feat(scripts): derive a stable ASCII clip id from a source filename"
```

---

## Task 2: Ingest one clip to a sample directory

**Files:**
- Modify: `scripts/ingest_real_clips.py`
- Test: `scripts/tests/test_ingest_real_clips.py`

**Interfaces:**
- Consumes: `clip_id_for` from Task 1; `scripts.screen_clean_clips.read_frames_lenient(path: Path, max_dim: int | None) -> tuple[list[np.ndarray], float, str]` (existing — returns RGB float32 frames in [0, 1], already downscaled so the long side is at most `max_dim`, using INTER_AREA).
- Produces: `ingest_clip(source: Path, output_dir: Path, max_dim: int, overwrite: bool = False) -> dict`. Returns the written `meta.json` contents. Creates `output_dir/<clip_id>/degraded/%06d.png` and `output_dir/<clip_id>/meta.json`.

- [ ] **Step 1: Write the failing test**

```python
# append to scripts/tests/test_ingest_real_clips.py
import json

import cv2
import numpy as np
import pytest

from scripts.ingest_real_clips import ingest_clip


def _write_video(path, frames, fps=30.0):
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def _synthetic_clip(tmp_path, n_frames, height=64, width=128, fps=30.0):
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (height, width, 3), dtype=np.uint8) for _ in range(n_frames)]
    path = tmp_path / "src.mp4"
    _write_video(path, frames, fps)
    return path


def test_ingest_clip_writes_frames_and_meta(tmp_path):
    source = _synthetic_clip(tmp_path, n_frames=40)
    out = tmp_path / "out"

    meta = ingest_clip(source, out, max_dim=64)

    sample_dir = out / meta["clip_id"]
    assert (sample_dir / "meta.json").exists()
    written = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    assert written == meta
    assert written["source"] == "src.mp4"
    assert written["fps"] > 0
    assert written["frames"] == len(list((sample_dir / "degraded").glob("*.png")))
    # Long side capped at max_dim.
    assert max(written["shape"]) == 64


def test_ingest_clip_frames_are_zero_padded_and_contiguous(tmp_path):
    source = _synthetic_clip(tmp_path, n_frames=40)
    out = tmp_path / "out"
    meta = ingest_clip(source, out, max_dim=64)
    degraded = out / meta["clip_id"] / "degraded"
    # MitigatorDataset._read_frame builds paths as f"{index:06d}.png" and
    # _read_frame_sequence assumes indices 0..n-1 with no gaps.
    for i in range(meta["frames"]):
        assert (degraded / f"{i:06d}.png").exists()


def test_ingest_clip_rejects_a_clip_too_short_to_survive_warmup(tmp_path):
    # The Detector's warm-up is one second, and Task 4 drops those frames
    # as training centres. A clip with fewer than round(fps) + 3 frames has
    # nothing left afterwards, so it must fail here rather than silently
    # contribute zero examples much later.
    source = _synthetic_clip(tmp_path, n_frames=12, fps=30.0)
    with pytest.raises(ValueError, match="too short"):
        ingest_clip(source, tmp_path / "out", max_dim=64)


def test_ingest_clip_skips_an_existing_sample_unless_overwritten(tmp_path):
    source = _synthetic_clip(tmp_path, n_frames=40)
    out = tmp_path / "out"
    meta = ingest_clip(source, out, max_dim=64)
    marker = out / meta["clip_id"] / "degraded" / "000000.png"
    marker.write_bytes(b"")  # corrupt it; a skip must leave this alone

    ingest_clip(source, out, max_dim=64)
    assert marker.read_bytes() == b""

    ingest_clip(source, out, max_dim=64, overwrite=True)
    assert marker.read_bytes() != b""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest scripts/tests/test_ingest_real_clips.py -q`
Expected: FAIL — `ImportError: cannot import name 'ingest_clip'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to scripts/ingest_real_clips.py
import json
import shutil

import cv2
import numpy as np

from scripts.screen_clean_clips import read_frames_lenient

# Matches mitigator/infer.py's _MAX_PROCESSING_DIM. Training and inference
# then see the same spatial frequencies; at native resolution a 256 patch
# is 3% of a 1920x1080 frame and 30% of a 360x640 one.
DEFAULT_MAX_DIM = 512


def ingest_clip(source: Path, output_dir: Path, max_dim: int, overwrite: bool = False) -> dict:
    clip_id = clip_id_for(source.name)
    sample_dir = output_dir / clip_id
    meta_path = sample_dir / "meta.json"

    if meta_path.exists() and not overwrite:
        return json.loads(meta_path.read_text(encoding="utf-8"))

    frames, fps, _note = read_frames_lenient(source, max_dim)
    minimum = round(fps) + 3
    if len(frames) < minimum:
        raise ValueError(
            f"{source.name}: too short -- {len(frames)} frames at {fps:.2f} fps, "
            f"need at least {minimum}. The Detector's warm-up is one second and "
            f"those frames are excluded as training centres, so nothing would remain."
        )

    degraded_dir = sample_dir / "degraded"
    if degraded_dir.exists():
        shutil.rmtree(degraded_dir)
    degraded_dir.mkdir(parents=True, exist_ok=True)

    for index, frame_rgb in enumerate(frames):
        # read_frames_lenient hands back RGB float32 in [0, 1];
        # MitigatorDataset._read_frame reads PNGs with cv2.imread (BGR uint8)
        # and converts back, so write BGR uint8 here. PNG is lossless, so the
        # round trip is exact.
        bgr = cv2.cvtColor(
            np.clip(frame_rgb * 255.0, 0, 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        if not cv2.imwrite(str(degraded_dir / f"{index:06d}.png"), bgr):
            raise OSError(f"failed to write frame {index} for {clip_id}")

    meta = {
        "clip_id": clip_id,
        "source": source.name,
        "fps": float(fps),
        "frames": len(frames),
        "shape": [int(frames[0].shape[0]), int(frames[0].shape[1])],
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
```

- [ ] **Step 4: Run test to verify it passes**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest scripts/tests/test_ingest_real_clips.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/ingest_real_clips.py scripts/tests/test_ingest_real_clips.py
git -c user.name="hoonseung2" -c user.email="dltmdgns0323@gmail.com" commit -m "feat(scripts): ingest a real flickering clip into the training sample layout"
```

---

## Task 3: Ingest CLI over a directory

**Files:**
- Modify: `scripts/ingest_real_clips.py`
- Test: `scripts/tests/test_ingest_real_clips.py`

**Interfaces:**
- Consumes: `ingest_clip` from Task 2.
- Produces: `main(argv: list[str] | None = None) -> int`, invoked as `python -m scripts.ingest_real_clips`. Flags: `--clips-dir`, `--output`, `--max-dim` (default 512), `--exclude` (repeatable, matched against the source *filename*), `--overwrite`.

- [ ] **Step 1: Write the failing test**

```python
# append to scripts/tests/test_ingest_real_clips.py
from scripts.ingest_real_clips import main


def test_main_ingests_every_clip_except_excluded_ones(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    for name in ("keep_a.mp4", "keep_b.mp4", "drop_me.mp4"):
        source = _synthetic_clip(tmp_path, n_frames=40)
        source.replace(clips / name)
    out = tmp_path / "out"

    exit_code = main([
        "--clips-dir", str(clips), "--output", str(out),
        "--max-dim", "64", "--exclude", "drop_me.mp4",
    ])

    assert exit_code == 0
    ingested = {
        json.loads((d / "meta.json").read_text(encoding="utf-8"))["source"]
        for d in out.iterdir() if d.is_dir()
    }
    assert ingested == {"keep_a.mp4", "keep_b.mp4"}


def test_main_errors_when_an_exclude_matches_nothing(tmp_path):
    # A typo'd exclusion would silently train on a clip meant to be dropped
    # -- including one excluded for evaluation-set contamination.
    clips = tmp_path / "clips"
    clips.mkdir()
    _synthetic_clip(tmp_path, n_frames=40).replace(clips / "keep.mp4")

    with pytest.raises(ValueError, match="matched no file"):
        main([
            "--clips-dir", str(clips), "--output", str(tmp_path / "out"),
            "--max-dim", "64", "--exclude", "typo.mp4",
        ])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest scripts/tests/test_ingest_real_clips.py -q`
Expected: FAIL — `ImportError: cannot import name 'main'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to scripts/ingest_real_clips.py
import argparse

_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest already-flickering clips into the MitigatorDataset layout."
    )
    parser.add_argument("--clips-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-dim", type=int, default=DEFAULT_MAX_DIM,
                        help="Cap the long side at this many pixels. Default matches "
                             "mitigator/infer.py's _MAX_PROCESSING_DIM.")
    parser.add_argument("--exclude", action="append", default=[],
                        help="Source filename to skip. Repeatable. Errors if it matches nothing.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    clips_dir = Path(args.clips_dir)
    sources = sorted(p for p in clips_dir.iterdir() if p.suffix.lower() in _VIDEO_SUFFIXES)

    names = {p.name for p in sources}
    for excluded in args.exclude:
        if excluded not in names:
            raise ValueError(
                f"--exclude {excluded!r} matched no file in {clips_dir}. "
                "A mistyped exclusion silently trains on a clip that was meant to be dropped."
            )
    sources = [p for p in sources if p.name not in set(args.exclude)]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen: dict[str, str] = {}
    for source in sources:
        meta = ingest_clip(source, output_dir, args.max_dim, overwrite=args.overwrite)
        clip_id = meta["clip_id"]
        if clip_id in seen:
            raise ValueError(f"clip_id collision: {clip_id!r} from {seen[clip_id]!r} and {source.name!r}")
        seen[clip_id] = source.name
        print(f"{clip_id}: {meta['frames']} frames at {meta['fps']:.2f} fps "
              f"({meta['shape'][1]}x{meta['shape'][0]}) <- {source.name}", flush=True)

    print(f"ingested {len(sources)} clips to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest scripts/tests/test_ingest_real_clips.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/ingest_real_clips.py scripts/tests/test_ingest_real_clips.py
git -c user.name="hoonseung2" -c user.email="dltmdgns0323@gmail.com" commit -m "feat(scripts): add the ingest CLI with explicit exclusions"
```

---

## Task 4: Risky frames without `meta["segments"]`

**Files:**
- Modify: `training/mitigator_dataset.py:167-176` (the `risky_indices` construction inside `MitigatorDataset.__init__`)
- Test: `training/tests/test_mitigator_dataset.py`

**Interfaces:**
- Consumes: `FramePrior.required_strength` (existing, `float`).
- Produces: no new public name. `MitigatorDataset` now accepts samples whose `meta.json` has no `segments` key.

- [ ] **Step 1: Write the failing test**

```python
# append to training/tests/test_mitigator_dataset.py
def test_dataset_derives_risky_frames_from_priors_when_meta_has_no_segments(tmp_path, monkeypatch):
    # Ingested real clips carry no "segments": the detection that would
    # produce them is the same detection compute_prior already runs, and
    # _assign_segment_strengths sets required_strength above 0 on exactly
    # the frames of a detected RiskSegment.
    import training.mitigator_dataset as dataset_module
    from prior.compute import FramePrior

    sample_dir = _write_sample(tmp_path, clip_id="realclip", n_frames=5, segments=None)
    priors = [
        FramePrior(frame_index=i, mask=np.ones((8, 8), dtype=bool),
                   required_strength=0.5 if i in (2, 3) else 0.0)
        for i in range(5)
    ]
    monkeypatch.setattr(dataset_module, "_load_or_compute_prior", lambda *a, **k: priors)

    ds = MitigatorDataset(tmp_path, _PROFILE, split="train",
                          split_map={"realclip": "train"})

    assert len(ds) == 2
```

Add this helper near the top of the test file if one does not already exist:

```python
def _write_sample(tmp_path, clip_id, n_frames, segments, fps=30.0):
    """Write a minimal sample directory: meta.json plus n_frames PNGs."""
    sample_dir = tmp_path / clip_id
    (sample_dir / "degraded").mkdir(parents=True)
    for i in range(n_frames):
        cv2.imwrite(str(sample_dir / "degraded" / f"{i:06d}.png"),
                    np.zeros((8, 8, 3), dtype=np.uint8))
    meta = {"clip_id": clip_id, "fps": fps, "frames": n_frames}
    if segments is not None:
        meta["segments"] = segments
    (sample_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return sample_dir
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest training/tests/test_mitigator_dataset.py -k no_segments -q`
Expected: FAIL — `KeyError: 'segments'` (and `TypeError` for the not-yet-added `split_map`, which Task 6 adds; if you are executing tasks strictly in order, temporarily drop the `split_map` argument and set `clip_id` to one the hash maps to `"train"`, then restore it in Task 6)

- [ ] **Step 3: Write minimal implementation**

Replace the `risky_indices` block:

```python
            # Synthetic samples carry the injected segments in meta.json.
            # Ingested real clips do not -- deriving them there would rerun
            # the detection compute_prior has already done. required_strength
            # is the same signal: prior/compute.py:_assign_segment_strengths
            # sets it above 0.0 on exactly the frames of a detected
            # RiskSegment, which is also the gate mitigate_segment uses.
            if "segments" in meta:
                risky_indices = set()
                for segment in meta["segments"]:
                    risky_indices.update(range(segment["start_frame"], segment["end_frame"] + 1))
            else:
                risky_indices = {p.frame_index for p in priors if p.required_strength > 0.0}
```

Note this must run *after* `priors = _load_or_compute_prior(...)`; move the block below that line if it currently sits above it.

- [ ] **Step 4: Run test to verify it passes**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest training/tests/test_mitigator_dataset.py -q`
Expected: PASS, with every pre-existing test in the file still passing

- [ ] **Step 5: Commit**

```bash
git add training/mitigator_dataset.py training/tests/test_mitigator_dataset.py
git -c user.name="hoonseung2" -c user.email="dltmdgns0323@gmail.com" commit -m "feat(training): derive risky frames from priors when meta has no segments"
```

---

## Task 5: Exclude the Detector warm-up from training centres

**Files:**
- Modify: `training/mitigator_dataset.py` (same `__init__` loop as Task 4)
- Test: `training/tests/test_mitigator_dataset.py`

**Interfaces:**
- Consumes: `meta["fps"]` (existing).
- Produces: no new public name.

- [ ] **Step 1: Write the failing test**

```python
# append to training/tests/test_mitigator_dataset.py
def test_dataset_excludes_the_first_second_as_training_centres(tmp_path, monkeypatch):
    # The Detector has no predecessor for frame 0, so it conservatively
    # masks the whole frame, and mask persistence ORs that forward. On real
    # clips whose risk segment starts at frame 0 this trains the model to
    # correct entire frames. uncertain_frames equals fps exactly on all 17
    # screened clips, so one second is the span to drop.
    import training.mitigator_dataset as dataset_module
    from prior.compute import FramePrior

    _write_sample(tmp_path, clip_id="realclip", n_frames=12, segments=None, fps=5.0)
    priors = [
        FramePrior(frame_index=i, mask=np.ones((8, 8), dtype=bool), required_strength=0.5)
        for i in range(12)
    ]
    monkeypatch.setattr(dataset_module, "_load_or_compute_prior", lambda *a, **k: priors)

    ds = MitigatorDataset(tmp_path, _PROFILE, split="train", split_map={"realclip": "train"})

    centres = sorted(index for _dir, index in ds._index)
    assert centres == [5, 6, 7, 8, 9, 10, 11]  # round(5.0) == 5 frames dropped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest training/tests/test_mitigator_dataset.py -k warm_up -q`
Expected: FAIL — `assert [0, 1, 2, ...] == [5, 6, ...]`

- [ ] **Step 3: Write minimal implementation**

Immediately after `risky_indices` is built, add:

```python
            # The frames the Detector could not judge are not evidence.
            # They stay in degraded/ so compute_prior still sees real
            # neighbours -- trimming them off the clip instead would just
            # give the trimmed clip its own warm-up one second later.
            warmup_frames = round(fps)
            risky_indices = {i for i in risky_indices if i >= warmup_frames}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest training/tests/test_mitigator_dataset.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/mitigator_dataset.py training/tests/test_mitigator_dataset.py
git -c user.name="hoonseung2" -c user.email="dltmdgns0323@gmail.com" commit -m "feat(training): drop the Detector warm-up second from training centres"
```

---

## Task 6: Explicit train/dev split map

**Files:**
- Modify: `training/mitigator_dataset.py:115` (`MitigatorDataset.__init__` signature and the `clip_split` call)
- Create: `configs/splits/phase2.json`
- Test: `training/tests/test_mitigator_dataset.py`

**Interfaces:**
- Consumes: `clip_split(clip_id: str) -> str` (existing — stays the default).
- Produces: `MitigatorDataset(data_dir, profile, split, split_map: dict[str, str] | None = None)`. Task 7 passes `split_map` in from a JSON file.

- [ ] **Step 1: Write the failing test**

```python
# append to training/tests/test_mitigator_dataset.py
def test_split_map_overrides_the_hash(tmp_path, monkeypatch):
    import training.mitigator_dataset as dataset_module
    from prior.compute import FramePrior

    for clip_id in ("alpha", "beta"):
        _write_sample(tmp_path, clip_id=clip_id, n_frames=6, segments=None, fps=1.0)
    priors = [
        FramePrior(frame_index=i, mask=np.ones((8, 8), dtype=bool), required_strength=0.5)
        for i in range(6)
    ]
    monkeypatch.setattr(dataset_module, "_load_or_compute_prior", lambda *a, **k: priors)
    split_map = {"alpha": "train", "beta": "val"}

    train = MitigatorDataset(tmp_path, _PROFILE, split="train", split_map=split_map)
    val = MitigatorDataset(tmp_path, _PROFILE, split="val", split_map=split_map)

    assert train.samples_scanned == 1
    assert val.samples_scanned == 1


def test_split_map_errors_when_it_disagrees_with_the_directory(tmp_path, monkeypatch):
    # The hash split needed no file kept in sync; an explicit map does.
    # Both directions are checked: a clip on disk but absent from the map
    # would be silently dropped from training, and a clip in the map but
    # absent from disk means the corpus is not what the map describes.
    import training.mitigator_dataset as dataset_module
    from prior.compute import FramePrior

    _write_sample(tmp_path, clip_id="alpha", n_frames=6, segments=None, fps=1.0)
    monkeypatch.setattr(
        dataset_module, "_load_or_compute_prior",
        lambda *a, **k: [FramePrior(frame_index=i, mask=np.ones((8, 8), dtype=bool),
                                    required_strength=0.5) for i in range(6)],
    )

    with pytest.raises(ValueError, match="missing from the split map"):
        MitigatorDataset(tmp_path, _PROFILE, split="train", split_map={"beta": "train"})

    with pytest.raises(ValueError, match="not found on disk"):
        MitigatorDataset(tmp_path, _PROFILE, split="train",
                         split_map={"alpha": "train", "ghost": "val"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest training/tests/test_mitigator_dataset.py -k split_map -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'split_map'`

- [ ] **Step 3: Write minimal implementation**

Change the signature and add validation before the scan loop:

```python
    def __init__(self, data_dir: Path, profile: ThresholdProfile, split: str,
                 split_map: dict[str, str] | None = None):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.data_dir = Path(data_dir)
        self._split_map = split_map
```

Add this helper method to the class:

```python
    def _split_of(self, clip_id: str) -> str:
        if self._split_map is None:
            return clip_split(clip_id)
        return self._split_map[clip_id]
```

Insert the two-way check immediately before the `for sample_dir in sorted(...)` loop:

```python
        if split_map is not None:
            on_disk = {
                json.loads((d / "meta.json").read_text(encoding="utf-8"))["clip_id"]
                for d in sorted(self.data_dir.iterdir())
                if (d / "meta.json").exists()
            }
            missing = sorted(on_disk - set(split_map))
            if missing:
                raise ValueError(
                    f"{len(missing)} clip(s) in {self.data_dir} are missing from the split map: "
                    f"{missing}. They would be silently dropped from both splits."
                )
            absent = sorted(set(split_map) - on_disk)
            if absent:
                raise ValueError(
                    f"{len(absent)} clip(s) in the split map were not found on disk: {absent}. "
                    f"The corpus is not what the map describes."
                )
```

Replace `if clip_split(meta["clip_id"]) != split:` with `if self._split_of(meta["clip_id"]) != split:`.

- [ ] **Step 4: Run test to verify it passes**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest training/tests/ -q`
Expected: PASS — every pre-existing test too, since `split_map` defaults to `None`

- [ ] **Step 5: Write the split file**

Generate it from the ingested corpus rather than typing ids by hand (Task 9 runs the ingestion; if `data/real` does not exist yet, do this step after Task 9 and commit then):

```bash
python - <<'PY'
import io, json
from pathlib import Path
from scripts.ingest_real_clips import clip_id_for

DEV = ["11832-233049403_medium.mp4", "녹음 2026-08-08 000241.mp4", "disco lights.mp4"]
mapping = {}
for d in sorted(Path("data/real").iterdir()):
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    mapping[meta["clip_id"]] = "val" if meta["source"] in DEV else "train"
assert sum(v == "val" for v in mapping.values()) == 3, mapping
Path("configs/splits").mkdir(parents=True, exist_ok=True)
io.open("configs/splits/phase2.json", "w", encoding="utf-8").write(
    json.dumps(mapping, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(mapping, indent=2, ensure_ascii=False))
PY
```

- [ ] **Step 6: Commit**

```bash
git add training/mitigator_dataset.py training/tests/test_mitigator_dataset.py configs/splits/phase2.json
git -c user.name="hoonseung2" -c user.email="dltmdgns0323@gmail.com" commit -m "feat(training): allow an explicit train/dev split map"
```

---

## Task 7: `--split-file` on the trainer

**Files:**
- Modify: `training/train_mitigator.py:306-307` (the two `MitigatorDataset(...)` constructions) and the argument parser
- Test: `training/tests/test_train_mitigator.py`

**Interfaces:**
- Consumes: `MitigatorDataset(..., split_map=...)` from Task 6.
- Produces: `--split-file PATH` on the CLI.

- [ ] **Step 1: Write the failing test**

```python
# append to training/tests/test_train_mitigator.py
def test_split_file_is_passed_through_to_both_datasets(tmp_path, monkeypatch):
    import training.train_mitigator as train_module

    seen = []

    class _Recorder(_FakeMitigatorDataset):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("split_map"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(train_module, "MitigatorDataset", _Recorder)
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps({"a": "train", "b": "val"}), encoding="utf-8")

    train_module.main([
        "--data-dir", str(tmp_path), "--profile", "configs/profiles/itu.json",
        "--checkpoint-dir", str(tmp_path / "ckpt"), "--epochs", "1",
        "--split-file", str(split_file),
    ])

    assert seen == [{"a": "train", "b": "val"}, {"a": "train", "b": "val"}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest training/tests/test_train_mitigator.py -k split_file -q`
Expected: FAIL — `SystemExit: 2` (unrecognised argument `--split-file`)

- [ ] **Step 3: Write minimal implementation**

Add the argument:

```python
    parser.add_argument("--split-file", default=None,
                        help="JSON mapping clip_id -> 'train' | 'val'. Omit to use the "
                             "clip_id hash (the synthetic path). Required for phase 2, where "
                             "the dev set has to span the difficulty range and 14 clips "
                             "cannot be hashed into one that does.")
```

And use it:

```python
    split_map = None
    if args.split_file:
        split_map = json.loads(Path(args.split_file).read_text(encoding="utf-8"))

    train_dataset = MitigatorDataset(Path(args.data_dir), profile, split="train", split_map=split_map)
    val_dataset = MitigatorDataset(Path(args.data_dir), profile, split="val", split_map=split_map)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest training/tests/test_train_mitigator.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/train_mitigator.py training/tests/test_train_mitigator.py
git -c user.name="hoonseung2" -c user.email="dltmdgns0323@gmail.com" commit -m "feat(training): add --split-file"
```

---

## Task 8: `--init-from`

**Files:**
- Modify: `training/train_mitigator.py` (argument parser and the `start_epoch` block around line 368)
- Test: `training/tests/test_train_mitigator.py`

**Interfaces:**
- Consumes: `save_checkpoint(path, model, optimizer, epoch, best_val_loss)` and `MitigatorNet` (existing).
- Produces: `--init-from PATH` on the CLI, and `load_model_weights(path: Path, model: MitigatorNet) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# append to training/tests/test_train_mitigator.py
def test_init_from_loads_weights_without_optimizer_or_epoch(tmp_path):
    from training.train_mitigator import load_model_weights

    torch.manual_seed(0)
    source = MitigatorNet()
    optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    dataloader = torch.utils.data.DataLoader(_FixedBatchDataset(), batch_size=4)
    train_one_epoch(source, optimizer, dataloader, device="cpu", profile=_PROFILE)
    path = tmp_path / "phase1.pt"
    save_checkpoint(path, source, optimizer, epoch=29, best_val_loss=0.17)

    target = MitigatorNet()
    load_model_weights(path, target)

    for a, b in zip(source.parameters(), target.parameters()):
        assert torch.equal(a, b)


def test_init_from_and_resume_together_is_an_error(tmp_path):
    import training.train_mitigator as train_module

    path = tmp_path / "phase1.pt"
    model = MitigatorNet()
    save_checkpoint(path, model, torch.optim.AdamW(model.parameters(), lr=1e-3), epoch=0)

    with pytest.raises(ValueError, match="mutually exclusive"):
        train_module.main([
            "--data-dir", str(tmp_path), "--profile", "configs/profiles/itu.json",
            "--checkpoint-dir", str(tmp_path / "ckpt"), "--epochs", "1",
            "--init-from", str(path), "--resume",
        ])


def test_init_from_reports_an_architecture_mismatch_clearly(tmp_path):
    # in_channels went from 9+1+16 to 9+1+1 once already, and the raw
    # load_state_dict error names a tensor shape rather than the cause.
    from training.train_mitigator import load_model_weights

    path = tmp_path / "wrong.pt"
    torch.save({"model": {"nope": torch.zeros(1)}, "epoch": 0}, path)

    with pytest.raises(ValueError, match="in_channels"):
        load_model_weights(path, MitigatorNet())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest training/tests/test_train_mitigator.py -k init_from -q`
Expected: FAIL — `ImportError: cannot import name 'load_model_weights'`

- [ ] **Step 3: Write minimal implementation**

```python
def load_model_weights(path: Path, model: MitigatorNet) -> None:
    """Load only the weights -- no optimizer state, no epoch counter.

    Distinct from load_checkpoint on purpose. --resume restores the epoch
    counter too, so pointing it at a finished run's checkpoint makes
    range(start_epoch, epochs) empty and the process exits successfully
    having trained nothing.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as exc:
        raise ValueError(
            f"{path} does not fit the current MitigatorNet. Check in_channels: it "
            f"changed from 9+1+16 to 9+1+1 when the histogram embedding was removed, "
            f"and checkpoints from before that cannot be loaded. Original error: {exc}"
        ) from exc
```

Argument and wiring:

```python
    parser.add_argument("--init-from", default=None,
                        help="Initialise the model from a checkpoint's weights only -- fresh "
                             "optimizer, epoch counter back to 0. Use this to start phase 2 "
                             "from a phase-1 checkpoint. Mutually exclusive with --resume.")
```

```python
    if args.init_from and args.resume:
        raise ValueError(
            "--init-from and --resume are mutually exclusive: --init-from starts a new run "
            "from borrowed weights, --resume continues an existing one."
        )
    ...
    if args.init_from:
        load_model_weights(Path(args.init_from), model)
        print(f"initialised weights from {args.init_from} (fresh optimizer, epoch 0)")
```

Place the `--init-from` load after `model.to(device)` and the optimizer construction, and leave `start_epoch = 0` untouched.

- [ ] **Step 4: Run test to verify it passes**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest training/tests/test_train_mitigator.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/train_mitigator.py training/tests/test_train_mitigator.py
git -c user.name="hoonseung2" -c user.email="dltmdgns0323@gmail.com" commit -m "feat(training): add --init-from for weights-only initialisation"
```

---

## Task 9: `--snapshot-every`

**Files:**
- Modify: `training/train_mitigator.py` (argument parser and the checkpoint-saving block around line 398)
- Test: `training/tests/test_train_mitigator.py`

**Interfaces:**
- Consumes: `save_checkpoint` (existing).
- Produces: `--snapshot-every N` on the CLI, writing `<checkpoint-dir>/epoch_%03d.pt`.

- [ ] **Step 1: Write the failing test**

```python
# append to training/tests/test_train_mitigator.py
def test_snapshot_every_keeps_intermediate_epochs(tmp_path, monkeypatch):
    # Phase 1 selected on val_loss and kept only best.pt and latest.pt, so
    # the epoch with the lowest soft_area (e22) could not be evaluated
    # afterwards at all. Snapshots are what make post-hoc selection possible.
    import training.train_mitigator as train_module

    monkeypatch.setattr(train_module, "MitigatorDataset", _FakeMitigatorDataset)
    checkpoint_dir = tmp_path / "ckpt"

    train_module.main([
        "--data-dir", str(tmp_path), "--profile", "configs/profiles/itu.json",
        "--checkpoint-dir", str(checkpoint_dir), "--epochs", "5",
        "--snapshot-every", "2",
    ])

    snapshots = sorted(p.name for p in checkpoint_dir.glob("epoch_*.pt"))
    assert snapshots == ["epoch_001.pt", "epoch_003.pt", "epoch_004.pt"]
```

The expected list is epochs 1 and 3 (every second epoch counting from 0) plus epoch 4, the final epoch, which is always snapshotted so the last state is never lost.

- [ ] **Step 2: Run test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest training/tests/test_train_mitigator.py -k snapshot -q`
Expected: FAIL — `SystemExit: 2` (unrecognised argument `--snapshot-every`)

- [ ] **Step 3: Write minimal implementation**

```python
    parser.add_argument("--snapshot-every", type=int, default=0,
                        help="Also keep <checkpoint-dir>/epoch_NNN.pt every N epochs (0 = off). "
                             "best.pt tracks val_loss, which has been measured not to track the "
                             "Detector verdict -- snapshots are what let a different epoch be "
                             "evaluated after the run instead of being lost.")
```

After the existing `save_checkpoint(latest_path, ...)` call:

```python
        if args.snapshot_every > 0 and (
            (epoch - start_epoch) % args.snapshot_every == 0 or epoch == args.epochs - 1
        ):
            save_checkpoint(checkpoint_dir / f"epoch_{epoch:03d}.pt", model, optimizer,
                            epoch, best_val_loss)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest training/tests/ scripts/tests/ -q`
Expected: PASS — the full 198-test baseline plus everything added here

- [ ] **Step 5: Commit**

```bash
git add training/train_mitigator.py training/tests/test_train_mitigator.py
git -c user.name="hoonseung2" -c user.email="dltmdgns0323@gmail.com" commit -m "feat(training): add --snapshot-every"
```

---

## Task 10: Run the ingestion and record the Tier 0 baselines

**Files:**
- Create: `data/real/` (gitignored output — check `.gitignore` covers it before committing anything else)
- Create: `docs/superpowers/specs/2026-08-08-phase2-dev-tier0-baselines.md`

**Interfaces:**
- Consumes: everything above.
- Produces: the ingested corpus and the three per-clip Tier 0 contrast figures the follow-up plan's selection constraint needs.

- [ ] **Step 1: Confirm `data/real/` will not be committed**

```bash
git check-ignore -v data/real 2>&1 || echo "NOT IGNORED -- add 'data/real/' to .gitignore and commit that first"
```

- [ ] **Step 2: Run the ingestion**

```bash
python -u -m scripts.ingest_real_clips --clips-dir data/raw/real_flicker --output data/real --max-dim 512 --exclude "7301-199291564_medium.mp4" --exclude "56426-479655491_medium.mp4" --exclude "1258-144566586_medium.mp4"
```

Expected: `ingested 14 clips to data/real`. If the count is not 14, stop — an exclusion did not match or a source file is missing.

- [ ] **Step 3: Verify the corpus totals**

```bash
python -c "
import json, io
from pathlib import Path
metas = [json.loads((d/'meta.json').read_text(encoding='utf-8')) for d in sorted(Path('data/real').iterdir()) if (d/'meta.json').exists()]
print('clips:', len(metas), 'frames:', sum(m['frames'] for m in metas))
print('max long side:', max(max(m['shape']) for m in metas))
"
```

Expected: 14 clips, 8190 frames, max long side 512.

- [ ] **Step 4: Write the split file**

Run the Task 6 Step 5 snippet now if it was deferred, then commit `configs/splits/phase2.json`.

- [ ] **Step 5: Record the dev-set Tier 0 baselines**

```bash
python -m fallback.cli --input "data/raw/real_flicker/11832-233049403_medium.mp4" --output /tmp/t0_dev1.mp4 --profile configs/profiles/itu.json
python -m scripts.evaluate_mitigation --original "data/raw/real_flicker/11832-233049403_medium.mp4" --mitigated /tmp/t0_dev1.mp4 --profile configs/profiles/itu.json
```

Repeat for `녹음 2026-08-08 000241.mp4` and `disco lights.mp4`. Record each clip's **in-segment mean contrast** before/after and the percentage change in the new spec file, one row per clip, alongside the remaining-segment count. Check `python -m fallback.cli --help` first — confirm the flag names before running.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-08-08-phase2-dev-tier0-baselines.md
git -c user.name="hoonseung2" -c user.email="dltmdgns0323@gmail.com" commit -m "docs: record Tier 0 baselines for the phase-2 dev clips"
```

---

## Self-Review

**Spec coverage.** §4 components: ingest script (Tasks 1–3), warm-up (Task 5), explicit split (Task 6), `--init-from` (Task 8), `--snapshot-every` (Task 9). §5 ingestion and clip_id (Tasks 1–3). §7 failure handling: utf-8 (Global Constraints and every read/write above), short clip (Task 2), collision (Tasks 1 and 3), `--init-from` + `--resume` (Task 8), architecture mismatch (Task 8), split map disagreement (Task 6). §8 tests: covered per task; the pure selection function is out of scope with `--eval-every`. §3 corpus: Task 10.

**Deliberately not covered**, per the agreed scope: `--eval-every`, in-process detector re-validation, and the selection function with the Tier 0 constraint. Task 10 Step 5 produces the baseline numbers that plan will need, so it is not blocked on this one.

**Not covered and worth noting:** the spec's §7 row "clip contributes 0 training examples → warn with the clip name" is not implemented here. `samples_scanned` / `samples_contributing` are already printed by `train_mitigator.main`, so the aggregate is visible; the per-clip warning is a small follow-up, not a blocker for starting training.

**Type consistency.** `clip_id_for(str) -> str` is used identically in Tasks 1, 3, and 6. `ingest_clip(source, output_dir, max_dim, overwrite)` returns the meta dict in Tasks 2, 3, and 10. `split_map: dict[str, str] | None` has the same name and type in Tasks 4, 5, 6, and 7. `load_model_weights(path, model) -> None` is defined and used only in Task 8.

**One ordering note:** Task 4's test uses `split_map`, which Task 6 adds. Task 4 Step 2 says how to run it before then. If executing with subagents, hand Tasks 4–6 to the same worker or run Task 6 first.
