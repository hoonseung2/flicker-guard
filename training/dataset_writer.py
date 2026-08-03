"""Writes an accepted SynthesizedSample to disk as a clean/degraded PNG
frame-sequence pair plus a metadata JSON, and supports resuming an
interrupted batch by skipping samples that already exist.

`meta.json` is written last, on purpose: `sample_exists` keys off it, so a
directory without it is an incomplete write that the next run redoes.
`_write_frames` therefore has to raise on a failed frame write rather than
let `cv2.imwrite`'s `False` return slip through — a truncated sample that
still got its `meta.json` would be skipped by every future resume run,
permanently.

Storage-cost tradeoff (accepted, not a bug): every (clip, profile, pattern)
combination stores its own full copy of the clean frames, so a clip run
against 6 profiles x 2 patterns holds 12 identical clean copies on disk.
That is deliberate — each sample directory stays self-contained and
independently loadable/movable, which is worth more here than the disk
saved by de-duplicating clean frames behind a shared store.
"""
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
        frame_path = out_dir / f"{i:06d}.png"
        # cv2.imwrite returns False instead of raising on disk-full,
        # permission failure or path-length overflow. Make that behave like a
        # crash so the sample never reaches its meta.json (see module docstring).
        if not cv2.imwrite(str(frame_path), frame_bgr_uint8):
            raise OSError(f"cv2.imwrite failed to write frame: {frame_path}")


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
