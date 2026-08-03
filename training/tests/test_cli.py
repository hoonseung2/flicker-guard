import json

import cv2
import numpy as np

from training.cli import main, run_batch

PROFILE_JSON = {
    "name": "tiny",
    "max_flashes_per_second": 3,
    "max_area_ratio": 0.10,
    "general_flash_dark_threshold": 0.80,
    "general_flash_delta_threshold": 0.10,
    "red_saturation_ratio_threshold": 0.80,
}


def _write_clip(path, n_frames=40, fps=10, size=(32, 32)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    frame = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()


def _write_profiles_dir(dir_path):
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "tiny.json").write_text(json.dumps(PROFILE_JSON), encoding="utf-8")


def test_run_batch_creates_accepted_samples(tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _write_clip(clips_dir / "bear.mp4")
    profiles_dir = tmp_path / "profiles"
    _write_profiles_dir(profiles_dir)
    out_dir = tmp_path / "out"

    summary = run_batch(clips_dir, profiles_dir, out_dir, samples_per_combo=1, seed=0)

    assert summary["accepted"] == 2  # general + red, one profile
    assert (out_dir / "bear__tiny__general__000" / "meta.json").exists()
    assert (out_dir / "bear__tiny__red__000" / "meta.json").exists()


def test_run_batch_skips_existing_samples_on_rerun(tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _write_clip(clips_dir / "bear.mp4")
    profiles_dir = tmp_path / "profiles"
    _write_profiles_dir(profiles_dir)
    out_dir = tmp_path / "out"

    run_batch(clips_dir, profiles_dir, out_dir, samples_per_combo=1, seed=0)
    second = run_batch(clips_dir, profiles_dir, out_dir, samples_per_combo=1, seed=1)

    assert second["accepted"] == 0
    assert second["skipped_existing"] == 2


def test_run_batch_records_unreadable_clip(tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "broken.mp4").write_bytes(b"not a real video")
    profiles_dir = tmp_path / "profiles"
    _write_profiles_dir(profiles_dir)
    out_dir = tmp_path / "out"

    summary = run_batch(clips_dir, profiles_dir, out_dir, samples_per_combo=1, seed=0)

    assert summary["accepted"] == 0
    assert summary["failed"] == 1
    assert summary["failed_details"][0]["reason"] == "unreadable_clip"


def test_main_writes_summary_json(tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _write_clip(clips_dir / "bear.mp4")
    profiles_dir = tmp_path / "profiles"
    _write_profiles_dir(profiles_dir)
    out_dir = tmp_path / "out"

    exit_code = main([
        "--clips-dir", str(clips_dir),
        "--profiles-dir", str(profiles_dir),
        "--output", str(out_dir),
    ])

    assert exit_code == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["accepted"] == 2
