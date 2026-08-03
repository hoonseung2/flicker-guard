"""One-off data-prep utility: pre-screen candidate "clean" source clips by
running the real Detector against them, unmodified, across every shipped
profile. A clip that already trips the Detector before any synthesis
happens is not a valid "clean" ground-truth source -- DatasetSynth's whole
correctness argument (byte-identical outside the injected window) breaks
down if the source itself already contains an undetected hazard.

Not part of the training/ package -- this is a manual vetting step for
whichever clips get pointed at training/cli.py's --clips-dir, run once per
candidate clip directory, not part of the tested pipeline itself.

Usage:
    python scripts/screen_clean_clips.py --clips-dir data/clips --profiles-dir configs/profiles
"""
import argparse
from pathlib import Path

from detector.cli import VideoReadError, read_video_frames
from detector.pipeline import run_detection
from detector.profiles import load_profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Screen candidate clean clips against the real Detector.")
    parser.add_argument("--clips-dir", required=True)
    parser.add_argument("--profiles-dir", default="configs/profiles")
    args = parser.parse_args(argv)

    profiles = [load_profile(p) for p in sorted(Path(args.profiles_dir).glob("*.json"))]

    clean, flagged = [], []
    for clip_path in sorted(Path(args.clips_dir).glob("*.mp4")):
        try:
            frame_iter, fps = read_video_frames(str(clip_path))
            frames = list(frame_iter)
        except VideoReadError as exc:
            print(f"{clip_path.name}: UNREADABLE ({exc})")
            flagged.append(clip_path.name)
            continue

        hits = []
        for profile in profiles:
            _, segments = run_detection(frames, fps=fps, profile=profile, margin_seconds=0.0)
            if segments:
                hits.append(profile.name)

        if hits:
            print(f"{clip_path.name}: FLAGGED under profile(s) {hits} -- not usable as a clean source")
            flagged.append(clip_path.name)
        else:
            print(f"{clip_path.name}: clean ({len(frames)} frames @ {fps:.1f}fps)")
            clean.append(clip_path.name)

    print(f"\n{len(clean)} clean, {len(flagged)} flagged, out of {len(clean) + len(flagged)} clips.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
