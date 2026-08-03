"""Batch driver: synthesizes PSE training samples from a directory of clean
clips against every profile JSON in --profiles-dir and both flash patterns,
writing accepted samples under --output and skipping ones that already
exist there (safe to re-run after an interrupted batch; --overwrite forces
regeneration).

Determinism: each sample's parameters come from an rng derived from
`(seed, sample_id)`, not from one shared stream threaded through the batch
loop. A shared stream makes every sample's parameters depend on how many
draws earlier samples happened to consume, so a resume run — where existing
combos are skipped and consume none — shifts the stream and re-issues
earlier draws to later sample ids, producing exact duplicate content under
different ids. Deriving per sample makes the output a pure function of
`(seed, sample_id)`: identical for a given id whether the batch ran straight
through, resumed, or grew via a larger --samples-per-combo.

Unlike `detector/cli.py`, which streams frames lazily, this tool
materializes each clip fully (`list(read_video_frames(...))`). That is
deliberate, not a regression of Detector's streaming fix: the retry loop
re-injects into the *same* frames with freshly sampled parameters up to 5
times, and each clip is reused across every profile x pattern combination,
both of which need random access to the whole clip. This is an offline
dataset-generation job with no latency budget, where re-decoding the clip
12+ times would cost far more than holding it in RAM.
"""
import argparse
import json
import zlib
from pathlib import Path

import numpy as np

from detector.cli import VideoReadError, read_video_frames
from detector.profiles import load_profile
from training.dataset_writer import (
    sample_exists,
    sample_id,
    sample_injection_mode,
    write_sample,
)
from training.params import ClipTooShortError
from training.synth import synthesize_sample, synthesize_sample_realistic

PATTERNS = ("general", "red")


def _iter_clip_paths(clips_dir: Path):
    return sorted(Path(clips_dir).glob("*.mp4"))


def _load_profiles(profiles_dir: Path) -> list:
    """Fail-fast on a missing/empty profiles directory (spec section 6:
    a broken or absent profile is a pre-batch configuration error). A
    malformed profile JSON already raises out of `load_profile`."""
    profiles_dir = Path(profiles_dir)
    if not profiles_dir.is_dir():
        raise FileNotFoundError(f"profiles directory does not exist: {profiles_dir}")
    profile_paths = sorted(profiles_dir.glob("*.json"))
    if not profile_paths:
        raise FileNotFoundError(f"profiles directory contains no *.json profiles: {profiles_dir}")
    return [load_profile(path) for path in profile_paths]


def _sample_rng(seed: int, sid: str) -> np.random.Generator:
    """Deterministic per-sample rng, a pure function of (seed, sample_id).

    `zlib.crc32` rather than the builtin `hash()`: `hash()` is salted per
    process for str, so it would not reproduce across runs.
    """
    return np.random.default_rng([seed, zlib.crc32(sid.encode("utf-8"))])


def _new_summary(profiles) -> dict:
    return {
        "clips_found": 0,
        "accepted": 0,
        "skipped_existing": 0,
        "failed": 0,
        # Per spec section 4: profile x pattern accepted/skipped counts, not
        # just global totals. Clip-level failures (an unreadable clip) are not
        # attributable to a profile x pattern, so they land in the global
        # `failed`/`failed_details` only.
        "by_profile_pattern": {
            f"{profile.name}/{pattern}": {"accepted": 0, "skipped_existing": 0, "failed": 0}
            for profile in profiles
            for pattern in PATTERNS
        },
        "failed_details": [],
    }


def _record_failure(
    summary: dict,
    reason: str,
    clip_id: str,
    profile_name: str | None = None,
    pattern: str | None = None,
    index: int | None = None,
    error: str | None = None,
) -> None:
    summary["failed"] += 1
    detail = {"clip_id": clip_id, "reason": reason}
    if profile_name is not None:
        detail["profile"] = profile_name
    if pattern is not None:
        detail["pattern"] = pattern
    if index is not None:
        detail["index"] = index
    if error is not None:
        detail["error"] = error
    summary["failed_details"].append(detail)
    if profile_name is not None and pattern is not None:
        summary["by_profile_pattern"][f"{profile_name}/{pattern}"]["failed"] += 1


def run_batch(
    clips_dir: Path,
    profiles_dir: Path,
    out_root: Path,
    samples_per_combo: int,
    seed: int = 0,
    overwrite: bool = False,
    injection_mode: str = "flat",
) -> dict:
    if injection_mode == "flat":
        synthesize = synthesize_sample
    elif injection_mode == "realistic":
        synthesize = synthesize_sample_realistic
    else:
        raise ValueError(f"unknown injection_mode: {injection_mode!r}")

    profiles = _load_profiles(profiles_dir)

    clips_dir = Path(clips_dir)
    if not clips_dir.is_dir():
        raise FileNotFoundError(f"clips directory does not exist: {clips_dir}")

    clip_paths = _iter_clip_paths(clips_dir)
    summary = _new_summary(profiles)
    # An existing-but-empty clips directory is not a config error the way a
    # missing profile is, but it must not read as a silent success either.
    summary["clips_found"] = len(clip_paths)

    for clip_path in clip_paths:
        clip_id = clip_path.stem
        try:
            frame_iter, fps = read_video_frames(str(clip_path))
            clean_frames = list(frame_iter)
        except VideoReadError as exc:
            _record_failure(summary, "unreadable_clip", clip_id, error=str(exc))
            continue

        for profile in profiles:
            for pattern in PATTERNS:
                combo = summary["by_profile_pattern"][f"{profile.name}/{pattern}"]
                for index in range(samples_per_combo):
                    sid = sample_id(clip_id, profile.name, pattern, index)
                    if not overwrite and sample_exists(out_root, sid):
                        existing_mode = sample_injection_mode(out_root, sid)
                        if existing_mode != injection_mode:
                            raise ValueError(
                                f"sample {sid!r} already exists under --output with "
                                f"injection_mode={existing_mode!r}, but this run "
                                f"requested injection_mode={injection_mode!r}. Pass "
                                f"--overwrite to regenerate it, or point --output at "
                                f"an empty directory."
                            )
                        summary["skipped_existing"] += 1
                        combo["skipped_existing"] += 1
                        continue
                    try:
                        sample = synthesize(
                            clean_frames, fps, profile, pattern, _sample_rng(seed, sid)
                        )
                    except ClipTooShortError as exc:
                        _record_failure(
                            summary, "clip_too_short", clip_id,
                            profile.name, pattern, index, str(exc),
                        )
                        continue
                    if sample is None:
                        _record_failure(
                            summary, "validation_exhausted", clip_id,
                            profile.name, pattern, index,
                        )
                        continue
                    write_sample(sample, out_root, clip_id, index, injection_mode=injection_mode)
                    summary["accepted"] += 1
                    combo["accepted"] += 1

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthesize PSE training samples from clean clips.")
    parser.add_argument("--clips-dir", required=True)
    parser.add_argument("--profiles-dir", default="configs/profiles")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-combo", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary-output")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="regenerate samples that already exist under --output instead of skipping them",
    )
    parser.add_argument(
        "--injection-mode",
        choices=["flat", "realistic"],
        default="flat",
        help=(
            "flat: overwrite pixels with a fixed color (original DatasetSynth "
            "behavior). realistic: multiply original luminance/color by a fixed "
            "gain in linear light space, preserving scene texture."
        ),
    )
    args = parser.parse_args(argv)

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = run_batch(
        clips_dir=Path(args.clips_dir),
        profiles_dir=Path(args.profiles_dir),
        out_root=out_root,
        samples_per_combo=args.samples_per_combo,
        seed=args.seed,
        overwrite=args.overwrite,
        injection_mode=args.injection_mode,
    )

    summary_path = Path(args.summary_output) if args.summary_output else out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
