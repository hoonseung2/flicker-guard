"""Standalone CLI: run the Detector on a video file and dump a JSON report
of per-frame scores and risk segments, independent of the Mitigator, so the
detector's thresholds can be tuned and validated on its own."""
import argparse
import dataclasses
import json

import cv2
import numpy as np

from detector.pipeline import run_detection
from detector.profiles import load_profile


def read_video_frames(path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(frame_rgb)
    cap.release()
    return frames, fps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the flicker-guard PSE detector on a video file.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--margin-seconds", type=float, default=0.5)
    args = parser.parse_args(argv)

    frames, fps = read_video_frames(args.video)
    profile = load_profile(args.profile)
    scores, segments = run_detection(frames, fps=fps, profile=profile, margin_seconds=args.margin_seconds)

    report = {
        "video": args.video,
        "profile": profile.name,
        "fps": fps,
        "scores": [dataclasses.asdict(s) for s in scores],
        "segments": [dataclasses.asdict(s) for s in segments],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
