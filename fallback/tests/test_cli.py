import json

import cv2
import numpy as np

from fallback.cli import main


def _write_flickering_video(path, n_frames=40, h=48, w=48, fps=20.0, flicker_start=10):
    # flicker_start=10 leaves a safe lead-in before the flash starts, for the
    # same reason fallback/tests/test_apply.py's
    # test_mitigate_with_fallback_clears_every_risk_segment does: a risk
    # segment that starts at frame 0 gives trailing_reference's causal window
    # zero prior context, so the reference mean itself jumps between frame 0
    # (a 1-sample window) and frame 1 (a 2-sample window) -- a real,
    # documented transition that no strength in [0, 1] can suppress. That is
    # an accepted limitation of mitigate_with_fallback, not something this
    # CLI-level fixture should be exercising.
    rng = np.random.default_rng(0)
    base = (rng.random((h, w, 3)) * 0.2 * 255).astype(np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for i in range(n_frames):
            frame = base.copy()
            if i >= flicker_start and i % 2 == 0:
                frame[:, : int(w * 0.6)] = 242
            writer.write(frame)
    finally:
        writer.release()


def test_cli_writes_a_video_and_a_report(tmp_path):
    source = tmp_path / "in.mp4"
    _write_flickering_video(source)
    output = tmp_path / "out.mp4"
    report = tmp_path / "report.json"

    exit_code = main([
        "--input", str(source),
        "--output", str(output),
        "--profile", "configs/profiles/itu.json",
        "--report", str(report),
    ])

    assert exit_code == 0
    assert output.exists()

    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["passed"] is True
    assert "segments" in data

    written = cv2.VideoCapture(str(output))
    try:
        assert int(written.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    finally:
        written.release()


def test_cli_returns_nonzero_when_the_clip_still_fails(tmp_path, monkeypatch):
    # The exit code is how a caller learns mitigation did not converge, so
    # it is asserted against a report that says so. Forcing a real clip to
    # fail would make this test depend on how well the escalation loop
    # happens to do on one fixture, which is test_apply.py's subject, not
    # the CLI's contract.
    import fallback.cli as cli_module
    from fallback.apply import FallbackReport, SegmentOutcome

    source = tmp_path / "in.mp4"
    _write_flickering_video(source)

    def _fake_mitigate(frames, fps, profile, max_rounds, strength_step):
        outcome = SegmentOutcome(
            start_frame=0, end_frame=9, initial_strength=0.5,
            final_strength=1.0, escalations=6, passed=False,
        )
        return list(frames), FallbackReport(
            segments=[outcome], passed=False, rounds=6, remaining_segments=1
        )

    monkeypatch.setattr(cli_module, "mitigate_with_fallback", _fake_mitigate)

    report = tmp_path / "report.json"
    exit_code = main([
        "--input", str(source),
        "--output", str(tmp_path / "out.mp4"),
        "--profile", "configs/profiles/itu.json",
        "--report", str(report),
    ])
    assert exit_code == 1
    assert json.loads(report.read_text(encoding="utf-8"))["passed"] is False


def test_cli_returns_two_and_writes_an_error_report_on_a_corrupt_input(tmp_path):
    # Mirrors detector/cli.py's contract (detector/tests/test_cli.py's
    # test_main_fails_loudly_on_corrupt_video): a decode failure must not
    # propagate as an unhandled traceback, and it must not be reported as if
    # nothing happened either. This is not hypothetical for this project --
    # two of the three real measurement clips' originals decode fewer frames
    # than their container declares and raise VideoReadError.
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"this is definitely not an mp4 container" * 32)
    report = tmp_path / "report.json"

    exit_code = main([
        "--input", str(corrupt),
        "--output", str(tmp_path / "out.mp4"),
        "--profile", "configs/profiles/itu.json",
        "--report", str(report),
    ])

    assert exit_code == 2
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["status"] == "error"
    assert "error" in data
    assert "segments" not in data
    assert not (tmp_path / "out.mp4").exists()
