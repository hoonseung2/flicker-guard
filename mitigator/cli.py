"""Batch CLI: reads a video file, runs every frame through mitigate_segment
(which already decides per-frame whether correction is needed and
mask-blends the result), and writes the corrected frames back out as a
video. Closes the gap between infer.py's frame-sequence API and an actual
video-in/video-out tool for manually evaluating a trained checkpoint on
real footage.

Audio is not preserved (decided out of scope) -- see
docs/superpowers/specs/2026-08-04-mitigator-video-cli-design.md. The whole
video is passed to mitigate_segment directly rather than pre-extracting
risky segments, since it already passes non-risky/no-target frames through
unchanged internally.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from detector.cli import read_video_frames
from detector.profiles import load_profile
from mitigator.arch import MitigatorNet
from mitigator.infer import mitigate_segment


def load_model(checkpoint_path: Path) -> MitigatorNet:
    model = MitigatorNet()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model


def write_video(frames: list[np.ndarray], path: Path, fps: float) -> None:
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    try:
        for frame_rgb in frames:
            frame_bgr_uint8 = cv2.cvtColor(
                np.clip(frame_rgb * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            )
            writer.write(frame_bgr_uint8)
    finally:
        writer.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a trained Mitigator checkpoint over a video file.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args(argv)

    frame_iter, fps = read_video_frames(args.input)
    frames = list(frame_iter)
    profile = load_profile(Path(args.profile))
    model = load_model(Path(args.checkpoint))

    corrected = mitigate_segment(frames, fps=fps, profile=profile, model=model)
    write_video(corrected, Path(args.output), fps)

    print(f"processed {len(frames)} frames -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
