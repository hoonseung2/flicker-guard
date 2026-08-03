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
