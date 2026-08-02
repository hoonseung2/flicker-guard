"""Standalone CLI: run the Detector on a video file and dump a JSON report
of per-frame scores and risk segments, independent of the Mitigator, so the
detector's thresholds can be tuned and validated on its own.

**MVP limitations — this tool does not certify content as safe.** See
`detector/pipeline.py` for the full list; the same text is emitted in every
report under the `caveats` key so a report can never be read out of context:

- flash counting is flagged-frame counting (~2x the visual flash rate), not
  full WCAG/Harding opposing-transition-pair analysis;
- `ThresholdProfile` encodes only a flash-frequency and a flagged-area limit,
  so Ofcom's dark-scene sub-rule and Japan's pattern-density rule are not
  detected (plan Task 5 deferred them; recorded here per finding I5);
- external validation (Harding FPA or equivalent) is required before
  production use, per README section 9.

Decoding failures are never reported as "no risk found": `read_video_frames`
raises `VideoReadError` if the file cannot be opened, decodes zero frames, or
decodes fewer frames than the container declares, and `main` then writes a
failure report and returns a nonzero exit code (README section 7 — a decode
failure routes to Fallback, it is not "safe"). Frames are yielded lazily, so
a long clip is never materialised in RAM.
"""
import argparse
import dataclasses
import json
import sys
from collections.abc import Iterator

import cv2
import numpy as np

from detector.pipeline import run_detection
from detector.profiles import load_profile

EXIT_OK = 0
EXIT_VIDEO_ERROR = 2

MVP_CAVEATS = [
    "MVP approximation of the WCAG/Harding General Flash and Red Flash "
    "Threshold Algorithm: flash counting is flagged-frame counting over a "
    "trailing 1s window (~2x the true visual flash rate, conservative), not "
    "full opposing-transition-pair analysis.",
    "Threshold profiles encode only a flash-frequency limit and a flagged-area "
    "limit. Ofcom's dark-scene sub-rule (luminance < 160 with contrast >= 20) "
    "and Japan's high-contrast pattern-density rule are NOT encoded and NOT "
    "detected.",
    "Motion compensation is global/pan-only; local object motion is not "
    "compensated and leaves a small residual flagged area at frame borders.",
    "Frames whose measurement window is incomplete (clip start) are reported "
    "with uncertain=true and treated conservatively as flagged.",
    "External validation (Harding FPA or equivalent) is required before "
    "production use. This report does not certify that content is safe.",
]


class VideoReadError(RuntimeError):
    """Raised when a video cannot be opened or decoded in full."""


def read_video_frames(path: str) -> tuple[Iterator[np.ndarray], float]:
    """Open `path` and return (lazy RGB float32 frame iterator, fps).

    The capture is opened eagerly so an unreadable file fails before any
    detection is attempted; frames themselves are yielded one at a time so a
    long clip never sits in RAM as a list (finding I6). The iterator raises
    `VideoReadError` at the end of the stream if nothing decoded or if fewer
    frames decoded than the container declared (finding C1) — a truncated
    decode must not be reported as a clean, risk-free scan.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoReadError(f"cannot open video file: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    declared_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def _iter_frames() -> Iterator[np.ndarray]:
        decoded = 0
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                decoded += 1
                yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        finally:
            cap.release()
        if decoded == 0:
            raise VideoReadError(f"decoded 0 frames from video file: {path}")
        if declared_frames and declared_frames > 0 and decoded < declared_frames:
            raise VideoReadError(
                f"truncated or corrupt video file: {path} "
                f"(decoded {decoded} of {int(declared_frames)} declared frames)"
            )

    return _iter_frames(), fps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the flicker-guard PSE detector on a video file.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--margin-seconds", type=float, default=0.5)
    args = parser.parse_args(argv)

    profile = load_profile(args.profile)
    try:
        frames, fps = read_video_frames(args.video)
        scores, segments = run_detection(
            frames, fps=fps, profile=profile, margin_seconds=args.margin_seconds
        )
    except VideoReadError as error:
        # Never emit a "no risk found" report for a file we could not read.
        report = {
            "video": args.video,
            "profile": profile.name,
            "status": "error",
            "error": str(error),
            "caveats": MVP_CAVEATS,
        }
        _write_report(args.output, report)
        print(f"error: {error}", file=sys.stderr)
        return EXIT_VIDEO_ERROR

    report = {
        "video": args.video,
        "profile": profile.name,
        "status": "ok",
        "fps": fps,
        "scores": [dataclasses.asdict(s) for s in scores],
        "segments": [dataclasses.asdict(s) for s in segments],
        "caveats": MVP_CAVEATS,
    }
    _write_report(args.output, report)
    return EXIT_OK


def _write_report(path: str, report: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
