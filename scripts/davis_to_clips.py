"""One-off data-prep utility: convert DAVIS JPEG frame sequences into .mp4
clips so they can be used as --clips-dir input to training/cli.py.

Not part of the training/ package (out of DatasetSynth's plan scope per
docs/superpowers/specs/2026-08-02-dataset-synth-design.md section 2 --
"DAVIS 다운로드 자동화는 이번 설계에서 제외"). This script is the manual
data-prep step that spec explicitly deferred.

DAVIS ships as JPEG sequences with no embedded frame-rate metadata, so this
encodes at a fixed, arbitrary fps (default 24, the commonly-cited
convention for DAVIS). The exact value doesn't affect correctness anywhere
downstream: training/cli.py reads the fps back out of the encoded mp4 via
cv2, so whatever we pick here simply becomes "the fps" for that clip as far
as the rest of the pipeline is concerned.

Usage:
    python scripts/davis_to_clips.py \
        --davis-root data/raw/DAVIS \
        --output data/clips \
        --fps 24 \
        --sequences bear camel dance-twirl   # omit to convert all
"""
import argparse
from pathlib import Path

import cv2


def convert_sequence(seq_dir: Path, out_path: Path, fps: float) -> int:
    frame_paths = sorted(seq_dir.glob("*.jpg"))
    if not frame_paths:
        return 0

    first = cv2.imread(str(frame_paths[0]))
    height, width = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            writer.write(frame)
    finally:
        writer.release()
    return len(frame_paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert DAVIS JPEG sequences to mp4 clips.")
    parser.add_argument("--davis-root", required=True, help="Path to extracted DAVIS/ directory")
    parser.add_argument("--output", required=True, help="Directory to write .mp4 files into")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--resolution", default="480p")
    parser.add_argument("--sequences", nargs="*", default=None, help="Specific sequence names; omit for all")
    args = parser.parse_args(argv)

    images_root = Path(args.davis_root) / "JPEGImages" / args.resolution
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    sequence_dirs = sorted(images_root.iterdir()) if args.sequences is None else [
        images_root / name for name in args.sequences
    ]

    converted = 0
    for seq_dir in sequence_dirs:
        if not seq_dir.is_dir():
            print(f"skip (not found): {seq_dir}")
            continue
        out_path = out_dir / f"{seq_dir.name}.mp4"
        n_frames = convert_sequence(seq_dir, out_path, args.fps)
        print(f"{seq_dir.name}: {n_frames} frames -> {out_path}")
        converted += 1

    print(f"\nConverted {converted} sequence(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
