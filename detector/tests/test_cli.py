import json
from pathlib import Path

import numpy as np
import cv2

from detector.cli import main

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
    assert "scores" in report
    assert "segments" in report
    assert len(report["scores"]) == 10
