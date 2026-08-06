"""Visual PriorCalc report: what the "illumination prior" stage produces.

PriorCalc is this project's take on BlazeBVD's stage-1 idea -- derive a
lighting prior classically instead of making the network learn it. For each
frame it emits the target illumination histogram the Mitigator is asked to
correct toward, plus the per-pixel mask marking what it may touch.

This renders both against the frame's own measured histogram, so the gap
the network has to close is visible: the measured distribution swings with
every flash, the target does not.

Run as a module from the repo root (scripts/ has no __init__.py):

    python -m scripts.prior_report --video clip.mp4 \
        --profile configs/profiles/kr.json --output prior.png
"""
import argparse
from pathlib import Path

import numpy as np

from detector.luminance import relative_luminance
from detector.profiles import load_profile
from prior.compute import DEFAULT_N_BINS, compute_prior
from prior.histogram import compute_illumination_histogram
from scripts.screen_clean_clips import read_frames_lenient


def _mean_brightness(histogram: np.ndarray, bins: np.ndarray) -> float:
    return float((histogram * bins).sum())


def render(measured, targets, mask_ratios, fps, out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    for candidate in ("Malgun Gothic", "NanumGothic", "DejaVu Sans"):
        try:
            plt.rcParams["font.family"] = candidate
            break
        except Exception:  # noqa: BLE001 - font probing, try the next
            continue
    plt.rcParams["axes.unicode_minus"] = False

    bg, panel = "#12151C", "#1D2431"
    amber, teal, text, muted = "#F0A202", "#3DBE9B", "#F2F4F8", "#9AA4B4"

    n = len(measured)
    times = np.arange(n) / fps
    bins = np.linspace(0, 1, DEFAULT_N_BINS + 1)[:-1] + 0.5 / DEFAULT_N_BINS

    fig = plt.figure(figsize=(13.2, 8.6), facecolor=bg)
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1.05], hspace=0.42,
                           left=0.075, right=0.975, top=0.875, bottom=0.075)

    # Shared colour scale so the two heatmaps are directly comparable.
    # A linear scale is unreadable on venue footage: most pixels are near
    # black, so one huge spike in the lowest bin flattens everything else to
    # background. Gamma-compress so the faint structure that actually moves
    # is visible.
    from matplotlib.colors import PowerNorm
    vmax = float(np.nanpercentile(np.concatenate([measured.ravel(), targets.ravel()]), 99.5))
    norm = PowerNorm(gamma=0.4, vmin=0.0, vmax=max(vmax, 1e-6))

    def heat(row, data, label, color):
        ax = fig.add_subplot(gs[row])
        ax.set_facecolor(panel)
        ax.imshow(data.T, aspect="auto", origin="lower", cmap="magma",
                  norm=norm, extent=(0.0, float(times[-1]), 0.0, 1.0))
        # Almost all mass sits below 0.4 on venue footage; showing the empty
        # top half would just shrink the part that carries the signal.
        ax.set_ylim(0.0, 0.4)
        ax.set_ylabel("밝기", color=muted, fontsize=10.5)
        ax.set_title(label, color=color, fontsize=13, fontweight="bold", loc="left", pad=6)
        ax.tick_params(colors=muted, labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_color("#39424F")
        return ax

    heat(0, measured, "① 실제 프레임의 밝기 분포 — 번쩍임에 따라 계속 흔들린다", amber)
    heat(1, np.nan_to_num(targets), "② PriorCalc가 만든 보정 목표 분포 — 안정적으로 유지된다", teal)

    ax = fig.add_subplot(gs[2])
    ax.set_facecolor(panel)
    meas_mean = np.array([_mean_brightness(h, bins) for h in measured])
    tgt_mean = np.array([_mean_brightness(t, bins) if not np.isnan(t).any() else np.nan
                         for t in targets])
    ax.plot(times, meas_mean, lw=1.5, color=amber, label="실제 프레임 평균 밝기")
    ax.plot(times, tgt_mean, lw=2.2, color=teal, label="PriorCalc 목표 평균 밝기")
    ax.set_xlim(0, times[-1])
    ax.set_xlabel("시간 (초)", color=muted, fontsize=11)
    ax.set_ylabel("평균 밝기", color=muted, fontsize=10.5)
    ax.tick_params(colors=muted, labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color("#39424F")
    ax.grid(color="#2A3140", lw=0.7, axis="y")
    legend = ax.legend(loc="upper right", facecolor=panel, edgecolor="#39424F", fontsize=10.5)
    for t in legend.get_texts():
        t.set_color(text)
    ax.set_title("③ 둘의 간격이 곧 신경망이 메워야 할 양", color=text,
                 fontsize=13, fontweight="bold", loc="left", pad=6)

    fig.suptitle(title, color=text, fontsize=17, fontweight="bold", x=0.02, ha="left", y=0.975)
    covered = float(np.mean(mask_ratios)) * 100
    fig.text(0.02, 0.915,
             f"보정 대상 마스크 평균 {covered:.1f}% — 목표는 마스크 바깥(번쩍이지 않는) 화소에서 뽑아 최근 1초로 평활화한다",
             color=muted, fontsize=11, ha="left")
    fig.savefig(out_path, dpi=150, facecolor=bg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render what PriorCalc produces for a video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", help="PNG path; omit to print the summary only")
    parser.add_argument(
        "--max-dim", type=int, default=None,
        help="downscale so the long side is at most this many pixels, for speed/memory",
    )
    args = parser.parse_args(argv)

    video_path = Path(args.video)
    frames, fps, note = read_frames_lenient(video_path, args.max_dim)
    profile = load_profile(Path(args.profile))
    priors = compute_prior(frames, fps=fps, profile=profile)

    bins = np.linspace(0, 1, DEFAULT_N_BINS + 1)[:-1] + 0.5 / DEFAULT_N_BINS
    measured = np.stack([
        compute_illumination_histogram(relative_luminance(f), n_bins=DEFAULT_N_BINS)
        for f in frames
    ])
    targets = np.stack([
        p.target_histogram if p.target_histogram is not None
        else np.full(DEFAULT_N_BINS, np.nan)
        for p in priors
    ])
    mask_ratios = [float(p.mask.mean()) for p in priors]

    have = sum(1 for p in priors if p.target_histogram is not None)
    meas_mean = np.array([_mean_brightness(h, bins) for h in measured])
    tgt_valid = np.array([_mean_brightness(t, bins) for t in targets if not np.isnan(t).any()])

    print(f"{video_path.name}  |  프로파일 {profile.name}  |  {len(frames)}프레임 @ {fps:.1f}fps")
    if note:
        print(f"  ! {note}")
    print(f"목표 분포를 만든 프레임: {have}/{len(priors)} ({have / len(priors):.1%})")
    print(f"보정 대상 마스크 면적: 평균 {np.mean(mask_ratios):.1%}  최대 {max(mask_ratios):.1%}")
    print(f"실제 평균 밝기  변동폭 {meas_mean.std():.4f}  (범위 {meas_mean.min():.3f}~{meas_mean.max():.3f})")
    if len(tgt_valid):
        print(f"목표 평균 밝기  변동폭 {tgt_valid.std():.4f}  (범위 {tgt_valid.min():.3f}~{tgt_valid.max():.3f})")
        print(f"  -> 목표가 실제보다 {meas_mean.std() / max(tgt_valid.std(), 1e-9):.1f}배 안정적")

    if args.output:
        render(measured, targets, mask_ratios, fps, Path(args.output),
               f"{video_path.name} — PriorCalc(조명 prior) 리포트 ({profile.name})")
        print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
