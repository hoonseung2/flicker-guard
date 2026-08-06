"""Correct a video using PriorCalc's output alone -- no neural network.

PriorCalc itself produces no image: it emits a target illumination
histogram and a repair mask per frame. This applies them the plainest way
possible, by histogram-matching the masked pixels onto the target, so that
"what does the prior alone buy us" has an answer you can watch.

Two uses:
  - It shows what PriorCalc's numbers mean in pixels.
  - It is the classical baseline the Mitigator has to beat, and the same
    idea as the Fallback (Tier 0) tier in README section 3, which is not
    built yet.

This is a diagnostic script, not part of the shipped pipeline.

Run as a module from the repo root (scripts/ has no __init__.py):

    python -m scripts.prior_only_correct --video clip.mp4 \
        --profile configs/profiles/kr.json --output corrected.mp4
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from detector.luminance import relative_luminance
from detector.profiles import load_profile
from prior.compute import DEFAULT_N_BINS, compute_prior
from scripts.screen_clean_clips import read_frames_lenient

_EPS = 1e-6


def match_to_target(luminance: np.ndarray, mask: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return `luminance` with the masked pixels re-mapped onto `target`.

    Classical histogram specification: walk the source CDF up to a level,
    then read back the level with the same rank in the target CDF. Only the
    masked pixels take part -- both in building the source histogram and in
    being remapped -- since the unmasked ones are the reference the target
    was derived from in the first place.
    """
    selected = luminance[mask]
    if selected.size == 0:
        return luminance

    edges = np.linspace(0.0, 1.0, DEFAULT_N_BINS + 1)
    centers = edges[:-1] + 0.5 / DEFAULT_N_BINS

    source_hist, _ = np.histogram(selected, bins=edges)
    source_cdf = np.cumsum(source_hist).astype(np.float64)
    source_cdf /= max(source_cdf[-1], _EPS)

    target_cdf = np.cumsum(target).astype(np.float64)
    target_cdf /= max(target_cdf[-1], _EPS)

    # Rank of each source level, then the target level holding that rank.
    ranks = np.interp(selected, centers, source_cdf)
    remapped = np.interp(ranks, target_cdf, centers)

    out = luminance.copy()
    out[mask] = remapped.astype(luminance.dtype)
    return out


def correct_frame(frame_rgb: np.ndarray, mask: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Scale each masked pixel's colour so its luminance lands on the target.

    Scaling all three channels by one gain keeps hue and saturation intact,
    which matters here: the point is to pull the brightness down, not to
    recolour the stage.
    """
    luminance = relative_luminance(frame_rgb)
    corrected = match_to_target(luminance, mask, target)
    gain = np.where(mask, corrected / np.maximum(luminance, _EPS), 1.0)
    return np.clip(frame_rgb * gain[..., None], 0.0, 1.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Correct a video with PriorCalc's target alone (no network).")
    parser.add_argument("--video", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True, help="MP4 path for the corrected video")
    parser.add_argument("--max-dim", type=int, default=None,
                        help="downscale so the long side is at most this many pixels")
    args = parser.parse_args(argv)

    video_path = Path(args.video)
    frames, fps, note = read_frames_lenient(video_path, args.max_dim)
    profile = load_profile(Path(args.profile))
    priors = compute_prior(frames, fps=fps, profile=profile)

    print(f"{video_path.name}  |  프로파일 {profile.name}  |  {len(frames)}프레임 @ {fps:.1f}fps")
    if note:
        print(f"  ! {note}")

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    corrected_count = 0
    before, after = [], []
    try:
        for prior in priors:
            frame = frames[prior.frame_index]
            if prior.target_histogram is not None and prior.mask.any():
                out = correct_frame(frame, prior.mask, prior.target_histogram)
                corrected_count += 1
            else:
                out = frame
            before.append(float(relative_luminance(frame).mean()))
            after.append(float(relative_luminance(out).mean()))
            writer.write(cv2.cvtColor((np.clip(out, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    before_arr, after_arr = np.array(before), np.array(after)
    jump_before = np.abs(np.diff(before_arr)).mean()
    jump_after = np.abs(np.diff(after_arr)).mean()
    print(f"보정한 프레임: {corrected_count}/{len(priors)}")
    print(f"프레임간 밝기 급변  보정 전 {jump_before:.5f}  →  보정 후 {jump_after:.5f}"
          f"  ({(1 - jump_after / max(jump_before, _EPS)):.1%} 감소)")
    print(f"밝기 변동폭(표준편차) 보정 전 {before_arr.std():.5f}  →  보정 후 {after_arr.std():.5f}"
          f"  ({(1 - after_arr.std() / max(before_arr.std(), _EPS)):.1%} 감소)")
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
