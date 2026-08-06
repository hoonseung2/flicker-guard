"""Visual detection report: run the Detector over a video and render where
the risk is and how much of each frame it covers.

detector/cli.py already dumps the same detection as JSON. This is the
look-at-it counterpart -- a timeline of per-frame flagged area with the
risk segments shaded over it, plus thumbnails sampled across the clip so
the peaks can be tied back to what was actually on screen.

Run as a module from the repo root (scripts/ has no __init__.py):

    python -m scripts.detection_report --video clip.mp4 \
        --profile configs/profiles/kr.json --output report.png
"""
import argparse
from pathlib import Path

import numpy as np

from detector.pipeline import run_detection_with_masks
from detector.profiles import load_profile
from scripts.screen_clean_clips import read_frames_lenient

_THUMBNAIL_COUNT = 8


def _timecode(seconds: float) -> str:
    return f"{int(seconds // 60):d}:{seconds % 60:05.2f}"


def summarize(segments, frame_count: int, fps: float) -> dict:
    risky_frames = set()
    for segment in segments:
        risky_frames.update(range(segment.start_frame, segment.end_frame + 1))
    return {
        "segments": len(segments),
        "risky_frames": len(risky_frames),
        "risky_ratio": len(risky_frames) / frame_count if frame_count else 0.0,
        "duration_seconds": frame_count / fps if fps else 0.0,
    }


def render(frames, fps, segments, area_ratios, uncertain, out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    for candidate in ("Malgun Gothic", "NanumGothic", "DejaVu Sans"):
        try:
            plt.rcParams["font.family"] = candidate
            break
        except Exception:  # noqa: BLE001 - font probing, any failure just tries the next
            continue
    plt.rcParams["axes.unicode_minus"] = False

    bg, panel = "#12151C", "#1D2431"
    amber, teal, text, muted = "#F0A202", "#3DBE9B", "#F2F4F8", "#9AA4B4"

    times = np.arange(len(frames)) / fps
    fig = plt.figure(figsize=(13.0, 7.0), facecolor=bg)
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1], hspace=0.32,
                           left=0.062, right=0.982, top=0.86, bottom=0.06)

    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(panel)
    for segment in segments:
        ax.axvspan(segment.start_frame / fps, segment.end_frame / fps,
                   color=amber, alpha=0.16, zorder=0)
    ax.plot(times, np.array(area_ratios) * 100, lw=1.6, color=amber)
    ax.set_xlim(0, times[-1] if len(times) else 1)
    # Frames with no predecessor are conservatively flagged whole (see
    # detector/pipeline.py); letting that set the scale flattens every real
    # detection to a sliver.
    measured = [r for r, u in zip(area_ratios, uncertain) if not u]
    ax.set_ylim(0, max(5.0, (max(measured) if measured else 0.0) * 100 * 1.25))
    ax.set_ylabel("검출된 화면 비율 (%)", color=muted, fontsize=11)
    ax.set_xlabel("시간 (초)", color=muted, fontsize=11)
    ax.tick_params(colors=muted, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#39424F")
    ax.grid(color="#2A3140", lw=0.7, axis="y")

    for segment in segments:
        start, end = segment.start_frame / fps, segment.end_frame / fps
        ax.annotate("", xy=(start, ax.get_ylim()[1] * 0.94), xytext=(end, ax.get_ylim()[1] * 0.94),
                    arrowprops=dict(arrowstyle="<->", color=amber, lw=1.4))
        ax.text((start + end) / 2, ax.get_ylim()[1] * 0.965,
                f"{_timecode(start)} ~ {_timecode(end)}",
                ha="center", va="bottom", color=amber, fontsize=9.5, fontweight="bold")

    # Thumbnails, evenly spaced, outlined when that frame is inside a segment.
    risky_frames = set()
    for segment in segments:
        risky_frames.update(range(segment.start_frame, segment.end_frame + 1))
    picks = np.linspace(0, len(frames) - 1, _THUMBNAIL_COUNT).astype(int)
    inner = gridspec.GridSpecFromSubplotSpec(1, _THUMBNAIL_COUNT, subplot_spec=gs[1], wspace=0.06)
    for slot, index in enumerate(picks):
        tax = fig.add_subplot(inner[slot])
        tax.imshow(np.clip(frames[index], 0, 1))
        tax.set_xticks([]); tax.set_yticks([])
        risky = index in risky_frames
        for spine in tax.spines.values():
            spine.set_color(amber if risky else "#39424F")
            spine.set_linewidth(2.2 if risky else 0.8)
        tax.set_xlabel(f"{_timecode(index / fps)}", color=amber if risky else muted, fontsize=9)

    fig.suptitle(title, color=text, fontsize=17, fontweight="bold", x=0.02, ha="left", y=0.975)
    fig.text(0.02, 0.905,
             "주황색 구간 = 위험 판정 · 선 = 그 프레임에서 위험으로 검출된 화면 비율",
             color=muted, fontsize=11, ha="left")
    fig.savefig(out_path, dpi=150, facecolor=bg)


def write_overlay(video_path: Path, masks, out_path: Path, fps: float) -> int:
    """Re-decode the video and write a copy with the flagged pixels tinted.

    Streams frame by frame rather than reusing the already-decoded list: at
    native resolution a 39s 720p clip is ~10 GB of float32 frames, and
    holding a second copy to draw on would double that.
    """
    import cv2

    tint = np.array([0.0, 165.0, 240.0], dtype=np.float32)  # BGR amber
    cap = cv2.VideoCapture(str(video_path))
    writer = None
    written = 0
    try:
        while written < len(masks):
            ok, frame_bgr = cap.read()
            if not ok:
                break
            mask = masks[written]
            if frame_bgr.shape[:2] != mask.shape:
                frame_bgr = cv2.resize(frame_bgr, (mask.shape[1], mask.shape[0]),
                                       interpolation=cv2.INTER_AREA)
            out = frame_bgr.astype(np.float32)
            out[mask] = 0.45 * out[mask] + 0.55 * tint
            if writer is None:
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (out.shape[1], out.shape[0]))
            writer.write(np.clip(out, 0, 255).astype(np.uint8))
            written += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render where a video's PSE risk is and how much it covers.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", help="PNG path; omit to print the summary only")
    parser.add_argument("--overlay", help="MP4 path: a copy of the video with flagged pixels tinted amber")
    parser.add_argument("--margin-seconds", type=float, default=0.5)
    parser.add_argument(
        "--max-dim", type=int, default=None,
        help="downscale so the long side is at most this many pixels, for speed. "
             "Omitted by default: downscaling changes what the Detector flags, and a "
             "report should show the detection the video actually gets.",
    )
    args = parser.parse_args(argv)

    video_path = Path(args.video)
    frames, fps, note = read_frames_lenient(video_path, args.max_dim)
    profile = load_profile(Path(args.profile))
    scores, segments, masks = run_detection_with_masks(
        frames, fps=fps, profile=profile, margin_seconds=args.margin_seconds
    )
    area_ratios = [float(mask.mean()) for mask in masks]
    stats = summarize(segments, len(frames), fps)

    print(f"{video_path.name}  |  프로파일 {profile.name}  |  {len(frames)}프레임 @ {fps:.1f}fps "
          f"({stats['duration_seconds']:.1f}초)")
    if note:
        print(f"  ! {note}")
    print(f"위험 구간 {stats['segments']}개  |  위험 프레임 {stats['risky_frames']}개 "
          f"({stats['risky_ratio']:.1%})")
    for i, segment in enumerate(segments, 1):
        start, end = segment.start_frame / fps, segment.end_frame / fps
        window = area_ratios[segment.start_frame:segment.end_frame + 1]
        peak = max(window) if window else 0.0
        print(f"  {i}. {_timecode(start)} ~ {_timecode(end)}  ({end - start:.1f}초)"
              f"   최대 검출 면적 {peak:.1%}")
    if not segments:
        print("  위험 구간 없음")
    measured = [r for r, s in zip(area_ratios, scores) if not s.uncertain]
    print(f"프레임당 검출 면적: 평균 {np.mean(measured):.1%}  최대 {max(measured):.1%}"
          f"   (측정 불가 프레임 {len(area_ratios) - len(measured)}개 제외)")

    if args.output:
        render(frames, fps, segments, area_ratios,
               [s.uncertain for s in scores], Path(args.output),
               f"{video_path.name} — 위험 검출 리포트 ({profile.name})")
        print(f"-> {args.output}")

    if args.overlay:
        written = write_overlay(video_path, masks, Path(args.overlay), fps)
        print(f"-> {args.overlay}  ({written}프레임, 주황색 = 위험 판정 화소)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
