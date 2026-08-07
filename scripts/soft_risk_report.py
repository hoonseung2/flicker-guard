"""How closely the differentiable relaxation tracks the Detector it stands
in for, measured on real footage rather than synthetic pairs.

`training.soft_risk.soft_flagged_area` exists to put the pass/fail rule
inside the loss. If it disagrees with `detector.flash` on real frames, the
loss optimises something adjacent to the rule instead of the rule -- which
is the exact failure the self-supervised design was written to remove, so
this disagreement is worth measuring before anything is trained on it.

Frames are streamed, never accumulated: these clips reach several GB as
float32 and the tool exists to be run on them.

    python -m scripts.soft_risk_report --video clip.mp4 \\
        --profile configs/profiles/itu.json
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from detector.flash import red_flash_mask, transition_mask
from detector.luminance import relative_luminance
from detector.motion import compensate_shift, estimate_global_shift
from detector.profiles import load_profile
from training.soft_risk import soft_flagged_area


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare soft_flagged_area against the Detector's own area.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--max-dim", type=int, default=480)
    args = parser.parse_args(argv)

    profile = load_profile(Path(args.profile))
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")

    prev_rgb = prev_luminance = None
    hard_areas, soft_areas = [], []
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            h, w = frame_bgr.shape[:2]
            scale = args.max_dim / max(h, w)
            if scale < 1.0:
                frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)),
                                       interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            luminance = relative_luminance(frame_rgb)

            if prev_rgb is not None:
                dx, dy = estimate_global_shift(prev_luminance, luminance)
                aligned_rgb = compensate_shift(prev_rgb, dx, dy)
                aligned_luminance = compensate_shift(prev_luminance, dx, dy)

                hard = transition_mask(
                    aligned_luminance, luminance,
                    dark_threshold=profile.general_flash_dark_threshold,
                    delta_threshold=profile.general_flash_delta_threshold,
                ) | red_flash_mask(
                    aligned_rgb, frame_rgb,
                    saturation_ratio_threshold=profile.red_saturation_ratio_threshold,
                )
                hard_areas.append(float(hard.mean()))

                to_t = lambda f: torch.from_numpy(f.transpose(2, 0, 1)).float()[None]
                with torch.no_grad():
                    soft_areas.append(
                        float(soft_flagged_area(to_t(aligned_rgb), to_t(frame_rgb),
                                                profile, tau=args.tau).item())
                    )
            prev_rgb, prev_luminance = frame_rgb, luminance
    finally:
        capture.release()

    if not hard_areas:
        raise SystemExit(f"{args.video}: fewer than two frames decoded")

    hard = np.array(hard_areas)
    soft = np.array(soft_areas)
    error = np.abs(soft - hard)
    limit = profile.max_area_ratio
    agree = ((soft > limit) == (hard > limit)).mean()

    print(f"{Path(args.video).name}  {len(hard)} pairs  tau={args.tau}")
    print(f"  mean |soft - hard|     {error.mean():.4f}")
    print(f"  p95  |soft - hard|     {np.percentile(error, 95):.4f}")
    print(f"  max  |soft - hard|     {error.max():.4f}")
    print(f"  verdict agreement      {agree:.1%}  (both sides of area > {limit})")
    print(f"  hard mean / soft mean  {hard.mean():.4f} / {soft.mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
