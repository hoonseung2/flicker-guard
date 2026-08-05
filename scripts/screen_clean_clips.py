"""One-off data-prep utility: pre-screen candidate "clean" source clips by
running the real Detector against them, unmodified, across every shipped
profile. A clip that already trips the Detector before any synthesis
happens is not a valid "clean" ground-truth source -- DatasetSynth's whole
correctness argument (byte-identical outside the injected window) breaks
down if the source itself already contains an undetected hazard.

This exists so candidate footage can be gathered in bulk without judging
"is this one flicker-free?" by eye: download plenty, screen them here, keep
whatever passes. Pass --output to have the passing clips copied out.

Not part of the training/ package -- this is a manual vetting step for
whichever clips get pointed at training/cli.py's --clips-dir, run once per
candidate clip directory, not part of the tested pipeline itself.

Run as a module from the repo root, not as a file path -- scripts/ has no
__init__.py, so `python scripts/screen_clean_clips.py` puts scripts/ rather
than the repo root on sys.path and the detector imports fail:

    python -m scripts.screen_clean_clips --clips-dir data/raw/candidates \
        --profiles-dir configs/profiles --output data/clips
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from detector.pipeline import run_detection
from detector.profiles import load_profile

VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi")

# Detection reads relative luminance and compares *ratios* (flagged area vs
# frame area), so it does not need native resolution -- and screening
# arbitrary stock footage at native 4K would mean gigabytes of decoded
# float32 frames per clip. 480 matches the DAVIS-derived clips the training
# set already uses.
DEFAULT_MAX_DIM = 480


def read_frames_lenient(path: Path, max_dim: int) -> tuple[list[np.ndarray], float, str]:
    """Decode `path` to RGB float32 frames, downscaled so the long side is
    at most `max_dim`. Returns (frames, fps, note).

    Deliberately does NOT use detector.cli.read_video_frames: that raises
    VideoReadError when a container declares more frames than actually
    decode, which is correct for the Detector CLI (a truncated decode must
    never be reported as a clean scan) but wrong here. Stock footage
    routinely declares a frame or two more than it holds -- a real 17s test
    clip declared 510 and decoded 507 -- and discarding otherwise-usable
    source footage over that would defeat the point of bulk screening. The
    mismatch is reported in `note` instead.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video file: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 0:  # 0, None, or NaN
            fps = 30.0
        declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        frames = []
        scale = None
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if scale is None:
                height, width = frame_bgr.shape[:2]
                scale = min(1.0, max_dim / max(height, width))
            if scale < 1.0:
                frame_bgr = cv2.resize(
                    frame_bgr,
                    (round(frame_bgr.shape[1] * scale), round(frame_bgr.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    finally:
        cap.release()

    if not frames:
        raise OSError(f"decoded 0 frames from video file: {path}")

    note = ""
    if declared and declared != len(frames):
        note = f"container declared {declared} frames, decoded {len(frames)}"
    return frames, fps, note


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Screen candidate clean clips against the real Detector.")
    parser.add_argument("--clips-dir", required=True)
    parser.add_argument("--profiles-dir", default="configs/profiles")
    parser.add_argument("--output", help="copy clips that pass every profile into this directory")
    parser.add_argument(
        "--max-dim", type=int, default=DEFAULT_MAX_DIM,
        help=f"downscale so the long side is at most this many pixels (default {DEFAULT_MAX_DIM})",
    )
    args = parser.parse_args(argv)

    profiles = [load_profile(p) for p in sorted(Path(args.profiles_dir).glob("*.json"))]
    if not profiles:
        raise ValueError(f"no profile JSON files found in {args.profiles_dir}")

    out_dir = Path(args.output) if args.output else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    clip_paths = sorted(
        p for p in Path(args.clips_dir).iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )
    if not clip_paths:
        print(f"no video files found in {args.clips_dir} (looked for {', '.join(VIDEO_SUFFIXES)})")
        return 0

    clean, flagged = [], []
    for clip_path in clip_paths:
        try:
            frames, fps, note = read_frames_lenient(clip_path, args.max_dim)
        except OSError as exc:
            print(f"{clip_path.name}: UNREADABLE ({exc})")
            flagged.append(clip_path.name)
            continue

        hits = []
        for profile in profiles:
            _, segments = run_detection(frames, fps=fps, profile=profile, margin_seconds=0.0)
            if segments:
                hits.append(profile.name)

        suffix = f" [{note}]" if note else ""
        if hits:
            print(f"{clip_path.name}: FLAGGED under {hits} -- not usable as a clean source{suffix}")
            flagged.append(clip_path.name)
        else:
            duration = len(frames) / fps
            print(f"{clip_path.name}: clean ({len(frames)} frames, {duration:.1f}s @ {fps:.1f}fps){suffix}")
            clean.append(clip_path.name)
            if out_dir is not None:
                shutil.copy2(clip_path, out_dir / clip_path.name)

    print(f"\n{len(clean)} clean, {len(flagged)} flagged, out of {len(clip_paths)} clips.")
    if out_dir is not None:
        print(f"copied {len(clean)} clean clip(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
