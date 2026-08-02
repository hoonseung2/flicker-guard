import json
import types
from pathlib import Path

import numpy as np
import cv2
import pytest

from detector.cli import VideoReadError, main, read_video_frames

PROFILE_PATH = Path(__file__).parent.parent.parent / "configs" / "profiles" / "kr.json"


def _write_test_video(path, n_frames=10, fps=10, size=(32, 32)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for i in range(n_frames):
        value = 10 if i % 2 == 0 else 245
        frame = np.full((size[1], size[0], 3), value, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_main_writes_json_report(tmp_path):
    video_path = tmp_path / "test.mp4"
    _write_test_video(video_path)
    output_path = tmp_path / "report.json"
    exit_code = main([
        "--video", str(video_path),
        "--profile", str(PROFILE_PATH),
        "--output", str(output_path),
    ])
    assert exit_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert "scores" in report
    assert "segments" in report
    assert len(report["scores"]) == 10


def test_report_scores_carry_the_renamed_windowed_fields(tmp_path):
    # I2/C2: the JSON contract exposes flagged-frame-count semantics and the
    # windowed area, not the old flash_count_last_second key.
    video_path = tmp_path / "test.mp4"
    _write_test_video(video_path)
    output_path = tmp_path / "report.json"
    assert main([
        "--video", str(video_path),
        "--profile", str(PROFILE_PATH),
        "--output", str(output_path),
    ]) == 0
    score = json.loads(output_path.read_text(encoding="utf-8"))["scores"][0]
    assert "flash_count_last_second" not in score
    assert set(score) == {
        "frame_index",
        "flagged_frame_count_last_second",
        "flagged_area_ratio",
        "max_flagged_area_in_window",
        "uncertain",
    }


def test_report_carries_mvp_caveats(tmp_path):
    # I5: the unencoded Ofcom/Japan rules and the external-validation
    # requirement must travel with every report.
    video_path = tmp_path / "test.mp4"
    _write_test_video(video_path)
    output_path = tmp_path / "report.json"
    assert main([
        "--video", str(video_path),
        "--profile", str(PROFILE_PATH),
        "--output", str(output_path),
    ]) == 0
    caveats = json.loads(output_path.read_text(encoding="utf-8"))["caveats"]
    joined = " ".join(caveats)
    assert "Ofcom" in joined and "Japan" in joined
    assert "Harding FPA" in joined


def test_main_fails_loudly_on_missing_video(tmp_path):
    # C1: cv2.VideoCapture does not raise; a missing file used to produce a
    # "no risk found" report and exit code 0.
    output_path = tmp_path / "report.json"
    exit_code = main([
        "--video", str(tmp_path / "does_not_exist.mp4"),
        "--profile", str(PROFILE_PATH),
        "--output", str(output_path),
    ])
    assert exit_code != 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "error"
    assert "scores" not in report and "segments" not in report


def test_main_fails_loudly_on_corrupt_video(tmp_path):
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"this is definitely not an mp4 container" * 32)
    output_path = tmp_path / "report.json"
    exit_code = main([
        "--video", str(corrupt),
        "--profile", str(PROFILE_PATH),
        "--output", str(output_path),
    ])
    assert exit_code != 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "error"
    assert "segments" not in report


def test_read_video_frames_raises_on_unopenable_file(tmp_path):
    with pytest.raises(VideoReadError):
        read_video_frames(str(tmp_path / "nope.mp4"))


def test_read_video_frames_raises_when_decode_is_truncated(tmp_path, monkeypatch):
    video_path = tmp_path / "test.mp4"
    _write_test_video(video_path, n_frames=10)

    real_capture = cv2.VideoCapture

    class _TruncatingCapture:
        """Decodes 4 frames then reports EOF, as a damaged stream would."""

        def __init__(self, path):
            self._inner = real_capture(path)
            self._served = 0

        def isOpened(self):
            return self._inner.isOpened()

        def get(self, prop):
            return self._inner.get(prop)

        def read(self):
            if self._served >= 4:
                return False, None
            self._served += 1
            return self._inner.read()

        def release(self):
            self._inner.release()

    monkeypatch.setattr(cv2, "VideoCapture", _TruncatingCapture)
    frames, fps = read_video_frames(str(video_path))
    assert fps == 10.0
    with pytest.raises(VideoReadError, match="truncated or corrupt"):
        list(frames)


def test_read_video_frames_yields_lazily(tmp_path):
    # I6: no materialised list — the reader must be a one-frame-at-a-time
    # iterator so a long clip never sits in RAM.
    video_path = tmp_path / "test.mp4"
    _write_test_video(video_path)
    frames, fps = read_video_frames(str(video_path))
    assert isinstance(frames, types.GeneratorType)
    assert not isinstance(frames, list)
    first = next(frames)
    assert first.dtype == np.float32
    assert first.shape == (32, 32, 3)
    assert 0.0 <= float(first.min()) and float(first.max()) <= 1.0
    assert len(list(frames)) == 9  # remaining frames still stream out
