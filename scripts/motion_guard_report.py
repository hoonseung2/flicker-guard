"""Side-by-side video showing what the discarded phase-correlation
confidence costs: the Detector's mask built with the shift as it is used
today, next to the same mask with a low-confidence shift rejected.

`detector.motion.estimate_global_shift` throws away the second value
`cv2.phaseCorrelate` returns -- the correlation peak height, i.e. how much
structure it actually locked onto. On footage without much structure (smoke,
dark stages, low contrast) that response collapses and the shift it returns
is noise, but `detector.pipeline` warps the previous frame by it anyway
before differencing. Measured on real clips: shifts up to 1103 pixels on a
480-pixel frame, with response 0.03-0.23 against 0.35-0.84 on sound
estimates.

This renders the consequence rather than describing it. Panels are the
source frame, the mask today, and the mask with the guard, with the
per-frame response and both shifts burned in.

Frames are streamed, never accumulated -- a 1000-frame 720p clip held as
float32 is several GB, and this tool exists to be run on exactly the long
concert footage that makes that a problem.

Run as a module from the repo root (scripts/ has no __init__.py):

    python -m scripts.motion_guard_report --video clip.mp4 \
        --profile configs/profiles/itu.json --output out.mp4
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from detector.flash import red_flash_mask, transition_mask
from detector.luminance import relative_luminance
from detector.motion import compensate_shift
from detector.profiles import ThresholdProfile, load_profile

_LABEL_HEIGHT = 30
_TINT = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # BGR: red
_TINT_STRENGTH = 0.55


def _mask_for(
    prev_rgb: np.ndarray,
    prev_luminance: np.ndarray,
    curr_rgb: np.ndarray,
    curr_luminance: np.ndarray,
    shift: tuple[float, float],
    profile: ThresholdProfile,
) -> np.ndarray:
    """The Detector's per-frame mask, for one choice of shift.

    Mirrors detector.pipeline._iter_scores_and_masks exactly -- compensate
    the previous frame, then OR the general-flash and red-flash tests -- so
    the two panels differ only in the shift they were handed.
    """
    dx, dy = shift
    aligned_luminance = compensate_shift(prev_luminance, dx, dy)
    aligned_rgb = compensate_shift(prev_rgb, dx, dy)
    return transition_mask(
        aligned_luminance,
        curr_luminance,
        dark_threshold=profile.general_flash_dark_threshold,
        delta_threshold=profile.general_flash_delta_threshold,
    ) | red_flash_mask(
        aligned_rgb, curr_rgb, saturation_ratio_threshold=profile.red_saturation_ratio_threshold
    )


def _tinted(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = frame_bgr.copy()
    out[mask] = out[mask] * (1.0 - _TINT_STRENGTH) + _TINT * _TINT_STRENGTH
    return out


def _panel(frame_bgr: np.ndarray, label: str) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    canvas = np.zeros((h + _LABEL_HEIGHT, w, 3), dtype=np.uint8)
    canvas[_LABEL_HEIGHT:] = np.clip(frame_bgr * 255.0, 0, 255).astype(np.uint8)
    cv2.putText(canvas, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the Detector's mask with and without a phase-correlation confidence guard."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.30,
                        help="Reject the shift when phaseCorrelate's response falls below this. "
                             "Measured populations: sound estimates 0.35-0.84, garbage 0.03-0.23.")
    parser.add_argument("--max-dim", type=int, default=480,
                        help="Downscale the long side before analysis. Mask area is "
                             "resolution-dependent, so this changes what is flagged; it is here "
                             "to keep long clips renderable, not to produce reportable numbers.")
    args = parser.parse_args(argv)

    profile = load_profile(Path(args.profile))
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    prev_rgb = prev_luminance = None
    guarded_count = 0
    n_pairs = 0
    worst = (0.0, 0.0)  # (raw shift magnitude, its response)

    try:
        while True:
            ok, frame_bgr_uint8 = capture.read()
            if not ok:
                break

            h, w = frame_bgr_uint8.shape[:2]
            scale = args.max_dim / max(h, w)
            if scale < 1.0:
                frame_bgr_uint8 = cv2.resize(
                    frame_bgr_uint8, ((int(w * scale) // 2) * 2, (int(h * scale) // 2) * 2),
                    interpolation=cv2.INTER_AREA,
                )

            frame_bgr = frame_bgr_uint8.astype(np.float32) / 255.0
            frame_rgb = frame_bgr[:, :, ::-1].copy()
            luminance = relative_luminance(frame_rgb)

            if prev_rgb is None:
                prev_rgb, prev_luminance = frame_rgb, luminance
                continue

            (dx, dy), response = cv2.phaseCorrelate(
                prev_luminance.astype(np.float32), luminance.astype(np.float32)
            )
            guarded = (0.0, 0.0) if response < args.threshold else (dx, dy)

            n_pairs += 1
            if response < args.threshold:
                guarded_count += 1
            if abs(dx) + abs(dy) > worst[0]:
                worst = (abs(dx) + abs(dy), response)

            raw_mask = _mask_for(prev_rgb, prev_luminance, frame_rgb, luminance, (dx, dy), profile)
            guarded_mask = _mask_for(prev_rgb, prev_luminance, frame_rgb, luminance, guarded, profile)

            row = np.hstack([
                _panel(frame_bgr, "SOURCE"),
                _panel(_tinted(frame_bgr, raw_mask), f"TODAY  shift ({dx:+.1f},{dy:+.1f})  area {raw_mask.mean():.3f}"),
                _panel(_tinted(frame_bgr, guarded_mask),
                       f"GUARDED  resp {response:.3f}{'  REJECTED' if response < args.threshold else ''}  area {guarded_mask.mean():.3f}"),
            ])

            if writer is None:
                writer = cv2.VideoWriter(
                    args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (row.shape[1], row.shape[0])
                )
            writer.write(row)

            prev_rgb, prev_luminance = frame_rgb, luminance
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if n_pairs == 0:
        raise SystemExit(f"{args.video}: fewer than two frames decoded")

    print(f"{args.video} -> {args.output}")
    print(f"  {n_pairs} frame pairs, {guarded_count} ({guarded_count / n_pairs:.1%}) had their shift rejected "
          f"at threshold {args.threshold}")
    print(f"  largest shift seen: {worst[0]:.1f}px at response {worst[1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
