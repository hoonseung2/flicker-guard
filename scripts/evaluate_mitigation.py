"""Before/after evaluation: run the Detector over the original and the
mitigated video and report whether the risk actually went away.

Every other metric we have -- val loss, PSNR against a synthetic clean
frame -- measures how close the output is to a ground truth we made up.
This one measures the thing the project is actually for: does our own
Detector still flag the output? Zero remaining risk segments is the bar.

The frame-difference statistics are secondary but worth having, because a
model can drive the segment count to zero by flattening the video into
mush, and a collapsed global contrast shows up here as a suspiciously
large drop in mean luminance delta.

Run as a module from the repo root (scripts/ has no __init__.py):

    python -m scripts.evaluate_mitigation --original in.mp4 \
        --mitigated out.mp4 --profile configs/profiles/itu.json
"""
import argparse
from pathlib import Path

import numpy as np

from detector.luminance import relative_luminance
from detector.pipeline import run_detection
from detector.profiles import load_profile
from detector.segments import _is_risky
from scripts.screen_clean_clips import read_frames_lenient


def _timecode(seconds: float) -> str:
    return f"{int(seconds // 60):d}:{seconds % 60:05.2f}"


def detection_stats(frames: list[np.ndarray], fps: float, profile) -> dict:
    scores, segments = run_detection(frames, fps=fps, profile=profile)

    risky_frames = set()
    for segment in segments:
        risky_frames.update(range(segment.start_frame, segment.end_frame + 1))

    # Segments already carry the +-0.5s margin padding, so counting frames
    # that trip _is_risky separately tells us how much of the flagged span
    # is real risk versus margin -- a correction that shrinks the risk to a
    # single frame still produces a full second of padded segment, and
    # without this the two would look identical.
    triggering = [s.frame_index for s in scores if _is_risky(s, profile)]

    return {
        "segments": segments,
        "segment_count": len(segments),
        "risky_frames": len(risky_frames),
        "risky_ratio": len(risky_frames) / len(frames) if frames else 0.0,
        "triggering_frames": len(triggering),
        "peak_area": max((s.max_flagged_area_in_window for s in scores), default=0.0),
        "peak_flashes": max((s.flagged_frame_count_last_second for s in scores), default=0),
    }


def luminance_stats(frames: list[np.ndarray]) -> dict:
    """Frame-to-frame mean luminance jumps -- the flicker as a viewer feels it.

    Reported alongside the detector verdict as a collapse check: if the
    mitigated clip's mean luminance itself dropped sharply, the model
    removed the flicker by removing the picture.
    """
    means = np.array([float(relative_luminance(f).mean()) for f in frames])
    deltas = np.abs(np.diff(means))
    return {
        "mean_delta": float(deltas.mean()) if deltas.size else 0.0,
        "p95_delta": float(np.percentile(deltas, 95)) if deltas.size else 0.0,
        "max_delta": float(deltas.max()) if deltas.size else 0.0,
        "mean_luminance": float(means.mean()) if means.size else 0.0,
    }


def _print_report(before: dict, after: dict, lum_before: dict, lum_after: dict, fps: float) -> None:
    def change(a: float, b: float) -> str:
        if a == 0:
            return "  --  "
        return f"{(b - a) / a * 100:+6.1f}%"

    print("\n=== Detector verdict (the bar: 0 segments) ===")
    print(f"{'metric':<28}{'before':>10}{'after':>10}{'change':>10}")
    for label, key in [
        ("risk segments", "segment_count"),
        ("frames inside a segment", "risky_frames"),
        ("frames tripping the rule", "triggering_frames"),
    ]:
        print(f"{label:<28}{before[key]:>10}{after[key]:>10}{change(before[key], after[key]):>10}")
    for label, key in [("peak windowed area", "peak_area"), ("peak flashes/s", "peak_flashes")]:
        print(f"{label:<28}{before[key]:>10.3f}{after[key]:>10.3f}{change(before[key], after[key]):>10}")

    print("\n=== Frame-to-frame luminance (collapse check) ===")
    print(f"{'metric':<28}{'before':>10}{'after':>10}{'change':>10}")
    for label, key in [
        ("mean delta", "mean_delta"),
        ("p95 delta", "p95_delta"),
        ("max delta", "max_delta"),
        ("mean luminance", "mean_luminance"),
    ]:
        print(f"{label:<28}{lum_before[key]:>10.4f}{lum_after[key]:>10.4f}"
              f"{change(lum_before[key], lum_after[key]):>10}")
    print("  (mean luminance should stay roughly flat -- a large drop means the\n"
          "   flicker was removed by darkening the video, not by repairing it)")

    print("\n=== Remaining risk segments ===")
    if not after["segments"]:
        print("  none")
    else:
        for segment in after["segments"]:
            start, end = segment.start_frame / fps, segment.end_frame / fps
            print(f"  {_timecode(start)} - {_timecode(end)}  "
                  f"(frames {segment.start_frame}-{segment.end_frame})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Detector output before and after mitigation.")
    parser.add_argument("--original", required=True)
    parser.add_argument("--mitigated", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--max-dim", type=int, default=None,
                        help="Optional downscale before analysis. Omit (default) to analyse "
                             "at native resolution -- area ratios are resolution-dependent, "
                             "so a downscale changes the verdict.")
    args = parser.parse_args(argv)

    profile = load_profile(Path(args.profile))
    original, fps = read_frames_lenient(Path(args.original), args.max_dim)[:2]
    mitigated = read_frames_lenient(Path(args.mitigated), args.max_dim)[0]

    if len(original) != len(mitigated):
        # Not fatal -- comparing the overlap is still informative -- but the
        # counts must be reported, because a silently truncated output would
        # otherwise look like a mitigation win.
        print(f"WARNING: frame count differs ({len(original)} vs {len(mitigated)}); "
              f"comparing the first {min(len(original), len(mitigated))} frames of each")
        n = min(len(original), len(mitigated))
        original, mitigated = original[:n], mitigated[:n]

    print(f"{len(original)} frames at {fps:.2f} fps ({len(original) / fps:.1f}s), profile {Path(args.profile).stem}")
    before = detection_stats(original, fps, profile)
    after = detection_stats(mitigated, fps, profile)
    _print_report(before, after, luminance_stats(original), luminance_stats(mitigated), fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
