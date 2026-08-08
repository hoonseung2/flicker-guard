"""Ingest already-flickering real clips into the layout MitigatorDataset reads.

No injection: these clips arrive degraded. Phase 1 synthesised its samples
by injecting flicker into clean sources, so this is a separate one-way
script rather than a flag on training/cli.py -- that CLI is built around
profiles x patterns x injection, and with injection off its --profiles-dir,
--samples-per-combo and --injection-mode arguments all become meaningless.
Its sample naming (clip__profile__pattern__000) has no meaning here either.

Run as a module from the repo root (scripts/ has no __init__.py):

    python -m scripts.ingest_real_clips \
        --clips-dir data/raw/real_flicker --output data/real --max-dim 512
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

from scripts.screen_clean_clips import read_frames_lenient

# Matches mitigator/infer.py's _MAX_PROCESSING_DIM. Training and inference
# then see the same spatial frequencies. It also keeps the patch crop
# meaningful across a corpus spanning 360x640 to 1920x1080: at native
# resolution a 256 patch is 3% of a 1080p frame and 30% of a 360x640 one.
DEFAULT_MAX_DIM = 512

_NON_SLUG = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_SLUG = 60
_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def clip_id_for(filename: str) -> str:
    """Stable ASCII directory name for a source clip.

    The corpus contains emoji, '#', '!', Korean and doubled spaces. Emoji in
    a directory name is risky on Windows, and clip_id is the key the
    train/dev split file is written against, so it has to be both ASCII and
    stable across runs. Folding everything outside [A-Za-z0-9._-] to '_' can
    map two different filenames onto one slug, so a hash of the *full
    original filename* is appended to keep them distinct.
    """
    slug = _NON_SLUG.sub("_", Path(filename).stem).strip("._")[:_MAX_SLUG] or "clip"
    digest = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"


def ingest_clip(source: Path, output_dir: Path, max_dim: int, overwrite: bool = False) -> dict:
    """Write one clip as <output_dir>/<clip_id>/{degraded/*.png, meta.json}."""
    clip_id = clip_id_for(source.name)
    sample_dir = output_dir / clip_id
    meta_path = sample_dir / "meta.json"

    if meta_path.exists() and not overwrite:
        return json.loads(meta_path.read_text(encoding="utf-8"))

    frames, fps, _note = read_frames_lenient(source, max_dim)
    minimum = round(fps) + 3
    if len(frames) < minimum:
        raise ValueError(
            f"{source.name}: too short -- {len(frames)} frames at {fps:.2f} fps, need at "
            f"least {minimum}. The Detector's warm-up is one second and those frames are "
            f"excluded as training centres, so this clip would contribute nothing."
        )

    degraded_dir = sample_dir / "degraded"
    if degraded_dir.exists():
        # A partial previous run would otherwise leave stale high-numbered
        # frames behind, and _read_frame_sequence counts *.png to decide how
        # many frames a sample has.
        shutil.rmtree(degraded_dir)
    degraded_dir.mkdir(parents=True, exist_ok=True)

    for index, frame_rgb in enumerate(frames):
        # read_frames_lenient returns RGB float32 in [0, 1];
        # MitigatorDataset._read_frame reads PNGs with cv2.imread (BGR uint8)
        # and converts back. PNG is lossless, so the round trip is exact.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest already-flickering clips into the MitigatorDataset layout."
    )
    parser.add_argument("--clips-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-dim", type=int, default=DEFAULT_MAX_DIM,
                        help="Cap the long side at this many pixels. The default matches "
                             "mitigator/infer.py's _MAX_PROCESSING_DIM so training and "
                             "inference see the same resolution.")
    parser.add_argument("--exclude", action="append", default=[],
                        help="Source filename to skip. Repeatable. Errors if it matches nothing.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate samples that already exist under --output.")
    args = parser.parse_args(argv)

    clips_dir = Path(args.clips_dir)
    sources = sorted(p for p in clips_dir.iterdir() if p.suffix.lower() in _VIDEO_SUFFIXES)

    names = {p.name for p in sources}
    for excluded in args.exclude:
        if excluded not in names:
            # Silence here would train on a clip that was meant to be
            # dropped -- one of the exclusions exists because its stock id is
            # adjacent to an evaluation clip.
            raise ValueError(
                f"--exclude {excluded!r} matched no file in {clips_dir}."
            )
    sources = [p for p in sources if p.name not in set(args.exclude)]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen: dict[str, str] = {}
    for source in sources:
        meta = ingest_clip(source, output_dir, args.max_dim, overwrite=args.overwrite)
        clip_id = meta["clip_id"]
        if clip_id in seen:
            raise ValueError(
                f"clip_id collision: {clip_id!r} from {seen[clip_id]!r} and {source.name!r}"
            )
        seen[clip_id] = source.name
        print(f"{clip_id}: {meta['frames']} frames at {meta['fps']:.2f} fps "
              f"({meta['shape'][1]}x{meta['shape'][0]})", flush=True)

    total = sum(
        json.loads((output_dir / cid / "meta.json").read_text(encoding="utf-8"))["frames"]
        for cid in seen
    )
    print(f"ingested {len(sources)} clips, {total} frames, to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
