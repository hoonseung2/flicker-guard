"""Batch CLI for the Tier 0 fallback: video in, video out, plus a JSON
report of the strength each segment needed.

The report is the point as much as the video is. The strengths it records
are the first concrete measurement of how much contrast has to be crushed
to clear the profile's area limit on real footage -- the number the neural
path has to beat, and the design input for replacing PriorCalc's target.

Audio is not preserved, matching mitigator/cli.py.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from detector.cli import read_video_frames
from detector.profiles import load_profile
from fallback.apply import mitigate_with_fallback
from mitigator.cli import write_video


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Tier 0 classical fallback over a video file."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--report", help="write a JSON report of per-segment strengths here")
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--strength-step", type=float, default=0.1)
    args = parser.parse_args(argv)

    frame_iter, fps = read_video_frames(args.input)
    frames = list(frame_iter)
    profile = load_profile(Path(args.profile))

    corrected, report = mitigate_with_fallback(
        frames,
        fps=fps,
        profile=profile,
        max_rounds=args.max_rounds,
        strength_step=args.strength_step,
    )
    write_video(corrected, Path(args.output), fps)

    payload = {
        "input": args.input,
        "output": args.output,
        "profile": profile.name,
        "frames": len(frames),
        "fps": fps,
        "passed": report.passed,
        "rounds": report.rounds,
        "remaining_segments": report.remaining_segments,
        "segments": [asdict(outcome) for outcome in report.segments],
    }
    if args.report:
        Path(args.report).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"processed {len(frames)} frames -> {args.output}")
    for outcome in report.segments:
        print(
            f"  segment {outcome.start_frame}-{outcome.end_frame}: "
            f"strength {outcome.initial_strength:.3f} -> {outcome.final_strength:.3f} "
            f"({outcome.rounds} escalations, {'passed' if outcome.passed else 'FAILED'})"
        )
    if report.passed:
        print("Detector re-validation: PASSED (0 risk segments)")
        return 0

    # A clip that still fails is not a crash, but it must not look like
    # success either -- README section 7 treats an exhausted escalation as
    # an incident to log, not something to swallow.
    print(f"Detector re-validation: FAILED ({report.remaining_segments} segments remain)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
