"""Batch driver: synthesizes PSE training samples from a directory of clean
clips against every profile JSON in --profiles-dir and both flash patterns,
writing accepted samples under --output and skipping ones that already
exist there (safe to re-run after an interrupted batch)."""
import argparse
import json
from pathlib import Path

import numpy as np

from detector.cli import VideoReadError, read_video_frames
from detector.profiles import load_profile
from training.dataset_writer import sample_exists, sample_id, write_sample
from training.synth import synthesize_sample

PATTERNS = ("general", "red")


def _iter_clip_paths(clips_dir: Path):
    return sorted(Path(clips_dir).glob("*.mp4"))


def run_batch(
    clips_dir: Path,
    profiles_dir: Path,
    out_root: Path,
    samples_per_combo: int,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    profile_paths = sorted(Path(profiles_dir).glob("*.json"))
    profiles = [load_profile(p) for p in profile_paths]

    summary = {"accepted": 0, "skipped_existing": 0, "failed": 0, "failed_details": []}

    for clip_path in _iter_clip_paths(clips_dir):
        clip_id = clip_path.stem
        try:
            frame_iter, fps = read_video_frames(str(clip_path))
            clean_frames = list(frame_iter)
        except VideoReadError as exc:
            summary["failed"] += 1
            summary["failed_details"].append(
                {"clip_id": clip_id, "reason": "unreadable_clip", "error": str(exc)}
            )
            continue

        for profile in profiles:
            for pattern in PATTERNS:
                for index in range(samples_per_combo):
                    sid = sample_id(clip_id, profile.name, pattern, index)
                    if sample_exists(out_root, sid):
                        summary["skipped_existing"] += 1
                        continue
                    try:
                        sample = synthesize_sample(clean_frames, fps, profile, pattern, rng)
                    except ValueError as exc:
                        summary["failed"] += 1
                        summary["failed_details"].append(
                            {"clip_id": clip_id, "profile": profile.name, "pattern": pattern,
                             "index": index, "reason": "clip_too_short", "error": str(exc)}
                        )
                        continue
                    if sample is None:
                        summary["failed"] += 1
                        summary["failed_details"].append(
                            {"clip_id": clip_id, "profile": profile.name, "pattern": pattern,
                             "index": index, "reason": "validation_exhausted"}
                        )
                        continue
                    write_sample(sample, out_root, clip_id, index)
                    summary["accepted"] += 1

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthesize PSE training samples from clean clips.")
    parser.add_argument("--clips-dir", required=True)
    parser.add_argument("--profiles-dir", default="configs/profiles")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-combo", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary-output")
    args = parser.parse_args(argv)

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = run_batch(
        clips_dir=Path(args.clips_dir),
        profiles_dir=Path(args.profiles_dir),
        out_root=out_root,
        samples_per_combo=args.samples_per_combo,
        seed=args.seed,
    )

    summary_path = Path(args.summary_output) if args.summary_output else out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
