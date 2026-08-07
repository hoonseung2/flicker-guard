"""Before/after evaluation: run the Detector over the original and the
mitigated video and report whether the risk actually went away.

Every other metric we have -- val loss, PSNR against a synthetic clean
frame -- measures how close the output is to a ground truth we made up.
This one measures the thing the project is actually for: does our own
Detector still flag the output? Zero remaining risk segments is the bar.

The frame-difference and contrast statistics are secondary but worth
having, because a model can drive the segment count to zero by flattening
the video into mush. Mean luminance barely moves when that happens -- a
correction that compresses each frame toward its own mean is close to
luminance-preserving by construction -- so within-frame contrast is
reported directly (contrast_stats) instead of being inferred indirectly
from luminance-delta drops.

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


def _frame_spreads(frames: list[np.ndarray]) -> np.ndarray:
    """Per-frame within-frame (spatial) luminance spread: one `relative_luminance`
    call and one `std` per frame.

    Callers that need more than one aggregate over overlapping frame lists
    (e.g. a whole-clip figure and an in-segment figure restricted to a
    subset of the same frames) should call this once and slice/index the
    result, then reduce each slice with `_aggregate_contrast`, rather than
    calling `contrast_stats` once per frame list -- that would recompute
    `relative_luminance` for any frame appearing in more than one list.
    """
    return np.array([float(relative_luminance(frame).std()) for frame in frames])


def _aggregate_contrast(spreads: np.ndarray) -> dict:
    """Reduce a per-frame spread array (from `_frame_spreads`) to the reported
    mean/p95/nan-count triple.

    NaN handling: a frame containing NaN pixels (the mitigator can pass
    NaN through when its model output is degenerate) produces a NaN std
    for that one frame. Aggregating with plain mean/percentile would let
    a single bad frame poison the entire report into "nan", hiding every
    other frame's real, informative contrast number. Instead this drops
    only the contaminated frame(s) from the aggregate (nanmean/nanpercentile)
    and, when any were dropped, that count is surfaced in the report rather
    than silently discarded.
    """
    finite = spreads[np.isfinite(spreads)]
    return {
        "mean_contrast": float(np.nanmean(finite)) if finite.size else 0.0,
        "p95_contrast": float(np.nanpercentile(finite, 95)) if finite.size else 0.0,
        "nan_frames": int(spreads.size - finite.size),
    }


def contrast_stats(frames: list[np.ndarray]) -> dict:
    """Within-frame luminance spread, averaged over the clip.

    This is the damage figure to report, not mean luminance. The Tier 0
    fallback compresses each frame toward its own trailing mean, so the
    mean is near-invariant by construction -- on one real clip it rose
    1.87% while contrast fell 52.8%. Reporting the mean as a cost makes a
    heavy correction look gentle; contrast is what actually moves.

    "Contrast" here is within-frame spatial spread (the std of a single
    frame's relative luminance), not frame-to-frame temporal contrast --
    that would just remeasure flicker under a different name. In particular
    this makes contrast, unlike a temporal measure, invariant to the order
    of the frames passed in: it is a per-frame quantity, reduced with an
    order-agnostic mean/percentile.

    This still computes its own `relative_luminance` per frame, independent
    of `luminance_stats`/`detection_stats`, which do the same thing on the
    same frames elsewhere in this module -- calling this function does not
    avoid that cross-function duplication. What it does avoid, via
    `_frame_spreads`/`_aggregate_contrast`, is recomputing luminance a
    second time for a frame that appears in more than one call to this
    function (e.g. a whole-clip call and an in-segment call over a subset
    of the same frames) -- callers with that shape should use those two
    helpers directly instead of calling this function more than once.

    NaN handling: see `_aggregate_contrast`. This defends against NaN
    pixels reaching this function from any source, not because a live
    passthrough path was found -- none was; the mitigator itself falls
    back to the original frame on non-finite model output before frames
    ever get here.
    """
    return _aggregate_contrast(_frame_spreads(frames))


def _print_report(before: dict, after: dict, lum_before: dict, lum_after: dict, fps: float,
                   contrast_before: dict, contrast_after: dict,
                   contrast_affected: tuple[dict, dict] | None) -> None:
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

    print("\n=== Within-frame contrast (the honest damage figure) ===")
    print(f"{'metric':<28}{'before':>10}{'after':>10}{'change':>10}")
    for label, key in [("whole-clip mean contrast", "mean_contrast"), ("whole-clip p95 contrast", "p95_contrast")]:
        print(f"{label:<28}{contrast_before[key]:>10.4f}{contrast_after[key]:>10.4f}"
              f"{change(contrast_before[key], contrast_after[key]):>10}")
    if contrast_before["nan_frames"] or contrast_after["nan_frames"]:
        print(f"  WARNING: {contrast_before['nan_frames']} before / {contrast_after['nan_frames']} after "
              f"frame(s) had NaN pixels and were dropped from the contrast average above (not from "
              f"any other section of this report).")

    # Whole-clip averaging silently dilutes the number that matters: a
    # correction can (and, on real footage, does) hammer contrast only
    # inside the frames it actually touches -- the segments flagged in the
    # ORIGINAL video -- while leaving the other 90%+ of the clip untouched.
    # Averaging over all frames then reports a small, easy-to-miss number
    # even when the local damage is severe (measured on a real clip: -6%
    # whole-clip vs -67% inside the flagged region). That is the same
    # failure this task exists to fix, one level removed -- so report the
    # in-segment number too, not just the diluted one, the same way the
    # detector verdict block above separates padded segment frames from
    # frames that actually tripped the rule.
    if contrast_affected is not None:
        contrast_before_affected, contrast_after_affected = contrast_affected
        print(f"{'in-segment mean contrast':<28}{contrast_before_affected['mean_contrast']:>10.4f}"
              f"{contrast_after_affected['mean_contrast']:>10.4f}"
              f"{change(contrast_before_affected['mean_contrast'], contrast_after_affected['mean_contrast']):>10}")
        print(f"{'in-segment p95 contrast':<28}{contrast_before_affected['p95_contrast']:>10.4f}"
              f"{contrast_after_affected['p95_contrast']:>10.4f}"
              f"{change(contrast_before_affected['p95_contrast'], contrast_after_affected['p95_contrast']):>10}")
        print("  (in-segment = only frames the ORIGINAL video had flagged as risky -- the region")
        print("   the correction actually touched. Whole-clip numbers dilute this if most of the")
        print("   clip was never at risk and so was never corrected.)")
        report_contrast_before = contrast_before_affected["mean_contrast"]
        report_contrast_after = contrast_after_affected["mean_contrast"]
    else:
        print("  (no risk segments were flagged on the original -- no in-segment figure to report)")
        report_contrast_before = contrast_before["mean_contrast"]
        report_contrast_after = contrast_after["mean_contrast"]

    contrast_change_pct = (
        (report_contrast_after - report_contrast_before) / report_contrast_before * 100
        if report_contrast_before else 0.0
    )
    print("  Tier 0 baseline on real clips: -52.8% (Anyma), -48.4% (Cera Khin), measured in-segment.")
    print(f"  This run: {contrast_change_pct:+.1f}% mean contrast"
          f"{' (in-segment)' if contrast_affected is not None else ' (whole-clip; no segment to isolate)'}.")
    print("  A neural correction that passes but costs more contrast than this loses to a")
    print("  classical method needing no GPU, no training data, and under 1 ms/frame.")

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

    # Per-frame spatial spread, computed once per clip. Both the whole-clip
    # and in-segment contrast figures below are reductions of these same
    # arrays (see _frame_spreads/_aggregate_contrast) -- not two separate
    # passes over relative_luminance for the frames that appear in both.
    original_spreads = _frame_spreads(original)
    mitigated_spreads = _frame_spreads(mitigated)

    # Frames the ORIGINAL video had flagged -- the region a correction
    # actually has reason to touch. Used to report contrast damage
    # in-segment as well as whole-clip; see the comment in _print_report.
    affected = sorted({
        i for segment in before["segments"]
        for i in range(segment.start_frame, segment.end_frame + 1)
        if i < len(original)
    })
    if affected:
        contrast_affected = (
            _aggregate_contrast(original_spreads[affected]),
            _aggregate_contrast(mitigated_spreads[affected]),
        )
    else:
        contrast_affected = None

    _print_report(
        before, after,
        luminance_stats(original), luminance_stats(mitigated),
        fps,
        _aggregate_contrast(original_spreads), _aggregate_contrast(mitigated_spreads),
        contrast_affected,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
