# Mitigator Video CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal CLI (`mitigator/cli.py`) that reads a video file, runs every frame through the existing `mitigate_segment` (which already decides per-frame whether correction is needed and mask-blends the result), and writes the corrected frames back out as a video file.

**Architecture:** A single new module reusing three already-shipped pieces verbatim: `detector.cli.read_video_frames` (video → frame iterator + fps), `mitigator.infer.mitigate_segment` (frame list → corrected frame list, self-contained risk detection + correction), and the `cv2.VideoWriter` pattern already used in `scripts/davis_to_clips.py` (frame list → `.mp4`). No changes to any existing file. Audio is not preserved (decided out of scope).

**Tech Stack:** Python 3.10+, opencv-python, torch (all already in requirements.txt). No new dependencies.

## Global Constraints

- No audio handling — video (frame) correction only.
- No new "extract risky segment" logic — `mitigate_segment` already self-determines per-frame whether to correct, so the whole video's frame list is passed to it directly.
- Reuse `detector.cli.read_video_frames`'s existing `VideoReadError` for decode failures — do not wrap or redefine it.
- Every new file follows the existing project style: no comments explaining *what* code does, only *why* where genuinely non-obvious; module docstring at the top; tests colocated in a `tests/` subpackage.

---

## File Structure

```
flicker-guard/
└── mitigator/
    ├── cli.py                 # Task 1 (new) — video-in/video-out batch CLI
    └── tests/
        └── test_cli.py        # Task 1 (new)
```

---

## Task 1: `mitigator/cli.py` — video-level batch CLI

**Files:**
- Create: `mitigator/cli.py`
- Test: `mitigator/tests/test_cli.py`

**Interfaces:**
- Consumes: `detector.cli.read_video_frames(path: str) -> tuple[Iterator[np.ndarray], float]`, `detector.profiles.load_profile(path: Path) -> ThresholdProfile`, `mitigator.arch.MitigatorNet`, `mitigator.infer.mitigate_segment(frames, fps, profile, model) -> list[np.ndarray]`, `training.train_mitigator.save_checkpoint` (test-only, to build a fixture checkpoint).
- Produces: `load_model(checkpoint_path: Path) -> MitigatorNet`, `write_video(frames: list[np.ndarray], path: Path, fps: float) -> None`, `main(argv: list[str] | None = None) -> int` (CLI entry point). Not consumed by any other module in this plan — this is a standalone user-facing tool.

- [ ] **Step 1: Write the failing tests**

Create `mitigator/tests/test_cli.py`:

```python
from pathlib import Path

import cv2
import numpy as np
import torch

from mitigator.arch import MitigatorNet
from mitigator.cli import load_model, main, write_video
from training.train_mitigator import save_checkpoint

SHIPPED_PROFILE = Path(__file__).parent.parent.parent / "configs" / "profiles" / "itu.json"


def _write_test_video(path: Path, n: int = 20, h: int = 16, w: int = 16, fps: float = 10.0) -> None:
    rng = np.random.default_rng(0)
    frame = (rng.uniform(0.1, 0.9, size=(h, w, 3)) * 255).astype(np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    try:
        for _ in range(n):
            writer.write(frame)
    finally:
        writer.release()


def test_load_model_restores_saved_weights(tmp_path):
    torch.manual_seed(0)
    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(checkpoint_path, model, optimizer, epoch=3)

    loaded = load_model(checkpoint_path)

    for original_param, loaded_param in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(original_param, loaded_param)


def test_write_video_round_trips_frame_count_and_resolution(tmp_path):
    frames = [np.full((12, 20, 3), 0.5, dtype=np.float32) for _ in range(7)]
    output_path = tmp_path / "out.mp4"

    write_video(frames, output_path, fps=15.0)

    cap = cv2.VideoCapture(str(output_path))
    try:
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 7
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 20
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 12
    finally:
        cap.release()


def test_write_video_pixel_values_survive_uint8_round_trip(tmp_path):
    frame = np.zeros((10, 10, 3), dtype=np.float32)
    frame[:, :, 0] = 0.2  # R
    frame[:, :, 1] = 0.6  # G
    frame[:, :, 2] = 0.9  # B
    output_path = tmp_path / "out.mp4"

    write_video([frame], output_path, fps=10.0)

    cap = cv2.VideoCapture(str(output_path))
    try:
        ok, bgr_uint8 = cap.read()
    finally:
        cap.release()
    assert ok
    rgb = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    assert np.allclose(rgb, frame, atol=0.02)  # mp4v/uint8 quantization tolerance


def test_main_runs_video_through_mitigate_segment_and_writes_output(tmp_path):
    input_path = tmp_path / "in.mp4"
    output_path = tmp_path / "out.mp4"
    checkpoint_path = tmp_path / "model.pt"
    _write_test_video(input_path, n=20, h=16, w=16, fps=10.0)

    torch.manual_seed(0)
    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    save_checkpoint(checkpoint_path, model, optimizer, epoch=0)

    exit_code = main([
        "--input", str(input_path),
        "--output", str(output_path),
        "--checkpoint", str(checkpoint_path),
        "--profile", str(SHIPPED_PROFILE),
    ])

    assert exit_code == 0
    assert output_path.exists()
    cap = cv2.VideoCapture(str(output_path))
    try:
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 20
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 16
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 16
    finally:
        cap.release()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest mitigator/tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mitigator.cli'`

- [ ] **Step 3: Implement `mitigator/cli.py`**

```python
"""Batch CLI: reads a video file, runs every frame through mitigate_segment
(which already decides per-frame whether correction is needed and
mask-blends the result), and writes the corrected frames back out as a
video. Closes the gap between infer.py's frame-sequence API and an actual
video-in/video-out tool for manually evaluating a trained checkpoint on
real footage.

Audio is not preserved (decided out of scope) -- see
docs/superpowers/specs/2026-08-04-mitigator-video-cli-design.md. The whole
video is passed to mitigate_segment directly rather than pre-extracting
risky segments, since it already passes non-risky/no-target frames through
unchanged internally.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from detector.cli import read_video_frames
from detector.profiles import load_profile
from mitigator.arch import MitigatorNet
from mitigator.infer import mitigate_segment


def load_model(checkpoint_path: Path) -> MitigatorNet:
    model = MitigatorNet()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    return model


def write_video(frames: list[np.ndarray], path: Path, fps: float) -> None:
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    try:
        for frame_rgb in frames:
            frame_bgr_uint8 = cv2.cvtColor(
                np.clip(frame_rgb * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            )
            writer.write(frame_bgr_uint8)
    finally:
        writer.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a trained Mitigator checkpoint over a video file.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args(argv)

    frame_iter, fps = read_video_frames(args.input)
    frames = list(frame_iter)
    profile = load_profile(Path(args.profile))
    model = load_model(Path(args.checkpoint))

    corrected = mitigate_segment(frames, fps=fps, profile=profile, model=model)
    write_video(corrected, Path(args.output), fps)

    print(f"processed {len(frames)} frames -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest mitigator/tests/test_cli.py -v`
Expected: all PASS

- [ ] **Step 5: Run the full repository test suite**

Run: `pytest -v` from the repo root.
Expected: all pass (pre-existing count + 4 new tests from this task).

- [ ] **Step 6: Commit**

```bash
git add mitigator/cli.py mitigator/tests/test_cli.py
git commit -m "feat(mitigator): add video-level batch CLI"
```

---

## Self-Review Notes (for the plan author, not a task)

- **Spec coverage:** Design doc section 3 (data flow) -> Task 1's `main()`. Section 4 (error handling: decode failure propagates `VideoReadError`, checkpoint mismatch propagates naturally) -> no special code needed, verified by inspection that neither `read_video_frames` nor `torch.load`/`load_state_dict` are wrapped. Section 5 (test strategy: synthetic frames + random-init model, no real trained checkpoint needed) -> all 4 tests use `tmp_path` and a freshly-constructed `MitigatorNet`, no real video or real trained weights required.
- **Known limitation carried forward from the design, not hidden:** this CLI runs the whole video through `mitigate_segment` in one pass rather than pre-extracting only the flagged segments — fine for a ~20s manual-evaluation clip (this plan's actual use case) but not an efficient pattern for long videos. Out of scope per the design doc's own framing (evaluation/demo tool, not the production Verifier/Fallback/BufferManager path).
