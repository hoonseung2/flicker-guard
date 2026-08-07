"""Real-clip verification. Excluded from the default run -- these decode
hundreds of full-resolution frames and re-run detection several times.

Run explicitly:  python -m pytest fallback/tests/test_real_footage.py -m slow -v -s
"""
from pathlib import Path

import numpy as np
import pytest

from detector.luminance import relative_luminance
from detector.pipeline import run_detection
from detector.profiles import load_profile
from fallback.apply import mitigate_with_fallback
from scripts.screen_clean_clips import read_frames_lenient

CLIP = Path("test_mp4/wonbon_640.mp4")


@pytest.mark.slow
@pytest.mark.skipif(not CLIP.exists(), reason=f"{CLIP} not present")
def test_real_concert_clip_passes_after_fallback():
    profile = load_profile(Path("configs/profiles/itu.json"))
    frames, fps, _note = read_frames_lenient(CLIP, None)

    _scores, before = run_detection(frames, fps=fps, profile=profile)
    assert before, "fixture clip is not risky -- the test would pass trivially"

    corrected, report = mitigate_with_fallback(frames, fps=fps, profile=profile)
    _scores, after = run_detection(corrected, fps=fps, profile=profile)

    def mean_luminance(clip):
        return float(np.mean([relative_luminance(f).mean() for f in clip]))

    def mean_contrast(clip):
        return float(np.mean([relative_luminance(f).std() for f in clip]))

    # Printed so the numbers land in the measurement document -- run with -s.
    print(f"\nsegments      {len(before)} -> {len(after)}")
    print(f"passed        {report.passed}   rounds {report.rounds}")
    for outcome in report.segments:
        print(
            f"  {outcome.start_frame}-{outcome.end_frame}: "
            f"{outcome.initial_strength:.3f} -> {outcome.final_strength:.3f} "
            f"({outcome.escalations} escalations)"
        )
    print(f"mean luminance {mean_luminance(frames):.4f} -> {mean_luminance(corrected):.4f}")
    print(f"mean contrast  {mean_contrast(frames):.4f} -> {mean_contrast(corrected):.4f}")

    assert after == []
    assert report.passed is True
