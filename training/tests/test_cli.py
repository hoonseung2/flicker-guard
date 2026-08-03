import json

import cv2
import numpy as np
import pytest

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


def _sample_bytes(sample_dir):
    return {
        path.relative_to(sample_dir).as_posix(): path.read_bytes()
        for path in sorted(sample_dir.rglob("*"))
        if path.is_file()
    }


def _standard_inputs(tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _write_clip(clips_dir / "bear.mp4")
    profiles_dir = tmp_path / "profiles"
    _write_profiles_dir(profiles_dir)
    return clips_dir, profiles_dir


def test_run_batch_output_is_identical_whether_or_not_earlier_samples_were_skipped(tmp_path):
    # Each sample's parameters must be a pure function of (seed, sample_id).
    # With a single shared rng stream, resume-skipped combos consumed no draws
    # and every later sample got a different (duplicated) draw instead.
    clips_dir, profiles_dir = _standard_inputs(tmp_path)

    straight = tmp_path / "straight"
    run_batch(clips_dir, profiles_dir, straight, samples_per_combo=2, seed=7)

    resumed = tmp_path / "resumed"
    run_batch(clips_dir, profiles_dir, resumed, samples_per_combo=1, seed=7)
    second = run_batch(clips_dir, profiles_dir, resumed, samples_per_combo=2, seed=7)

    assert second["skipped_existing"] == 2
    assert second["accepted"] == 2

    for sid in (
        "bear__tiny__general__000", "bear__tiny__general__001",
        "bear__tiny__red__000", "bear__tiny__red__001",
    ):
        assert _sample_bytes(straight / sid) == _sample_bytes(resumed / sid), sid


def test_run_batch_does_not_duplicate_synthesized_samples_within_a_run(tmp_path):
    clips_dir, profiles_dir = _standard_inputs(tmp_path)
    out_dir = tmp_path / "out"
    run_batch(clips_dir, profiles_dir, out_dir, samples_per_combo=3, seed=7)

    windows = [
        json.loads((out_dir / f"bear__tiny__general__{i:03d}" / "meta.json").read_text(encoding="utf-8"))[
            "injected_window"
        ]
        for i in range(3)
    ]
    assert len({w["start_frame"] for w in windows}) > 1


def test_run_batch_raises_when_profiles_dir_is_missing(tmp_path):
    clips_dir, _ = _standard_inputs(tmp_path)
    with pytest.raises(FileNotFoundError):
        run_batch(clips_dir, tmp_path / "nope", tmp_path / "out", samples_per_combo=1, seed=0)


def test_run_batch_raises_when_profiles_dir_has_no_profiles(tmp_path):
    clips_dir, _ = _standard_inputs(tmp_path)
    empty = tmp_path / "empty-profiles"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        run_batch(clips_dir, empty, tmp_path / "out", samples_per_combo=1, seed=0)


def test_run_batch_raises_when_clips_dir_is_missing(tmp_path):
    _, profiles_dir = _standard_inputs(tmp_path)
    with pytest.raises(FileNotFoundError):
        run_batch(tmp_path / "nope", profiles_dir, tmp_path / "out", samples_per_combo=1, seed=0)


def test_run_batch_reports_zero_clips_found_instead_of_silently_succeeding(tmp_path):
    _, profiles_dir = _standard_inputs(tmp_path)
    empty_clips = tmp_path / "empty-clips"
    empty_clips.mkdir()

    summary = run_batch(empty_clips, profiles_dir, tmp_path / "out", samples_per_combo=1, seed=0)

    assert summary["clips_found"] == 0
    assert summary["accepted"] == 0


def test_run_batch_reports_clips_found_count(tmp_path):
    clips_dir, profiles_dir = _standard_inputs(tmp_path)
    summary = run_batch(clips_dir, profiles_dir, tmp_path / "out", samples_per_combo=1, seed=0)
    assert summary["clips_found"] == 1


def test_run_batch_records_clip_too_short(tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _write_clip(clips_dir / "stub.mp4", n_frames=8)
    profiles_dir = tmp_path / "profiles"
    _write_profiles_dir(profiles_dir)

    summary = run_batch(clips_dir, profiles_dir, tmp_path / "out", samples_per_combo=1, seed=0)

    assert summary["accepted"] == 0
    assert summary["failed"] == 2  # general + red
    assert [d["reason"] for d in summary["failed_details"]] == ["clip_too_short", "clip_too_short"]
    assert summary["by_profile_pattern"]["tiny/general"]["failed"] == 1


def test_run_batch_overwrite_regenerates_existing_samples(tmp_path):
    clips_dir, profiles_dir = _standard_inputs(tmp_path)
    out_dir = tmp_path / "out"

    run_batch(clips_dir, profiles_dir, out_dir, samples_per_combo=1, seed=0)
    meta_path = out_dir / "bear__tiny__general__000" / "meta.json"
    meta_path.write_text(json.dumps({"stale": True}), encoding="utf-8")

    summary = run_batch(clips_dir, profiles_dir, out_dir, samples_per_combo=1, seed=0, overwrite=True)

    assert summary["skipped_existing"] == 0
    assert summary["accepted"] == 2
    assert json.loads(meta_path.read_text(encoding="utf-8"))["clip_id"] == "bear"


def test_main_overwrite_flag_bypasses_the_resume_skip(tmp_path):
    clips_dir, profiles_dir = _standard_inputs(tmp_path)
    out_dir = tmp_path / "out"
    argv = [
        "--clips-dir", str(clips_dir),
        "--profiles-dir", str(profiles_dir),
        "--output", str(out_dir),
    ]

    assert main(argv) == 0
    assert json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))["skipped_existing"] == 0
    assert main(argv) == 0
    assert json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))["skipped_existing"] == 2

    assert main(argv + ["--overwrite"]) == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["skipped_existing"] == 0
    assert summary["accepted"] == 2


def test_run_batch_summary_reports_per_profile_pattern_breakdown(tmp_path):
    clips_dir, profiles_dir = _standard_inputs(tmp_path)
    out_dir = tmp_path / "out"

    first = run_batch(clips_dir, profiles_dir, out_dir, samples_per_combo=1, seed=0)
    assert first["by_profile_pattern"]["tiny/general"] == {
        "accepted": 1, "skipped_existing": 0, "failed": 0,
    }
    assert first["by_profile_pattern"]["tiny/red"] == {
        "accepted": 1, "skipped_existing": 0, "failed": 0,
    }
    # the global totals stay alongside the breakdown
    assert first["accepted"] == 2

    second = run_batch(clips_dir, profiles_dir, out_dir, samples_per_combo=1, seed=0)
    assert second["by_profile_pattern"]["tiny/general"] == {
        "accepted": 0, "skipped_existing": 1, "failed": 0,
    }


def test_run_batch_realistic_mode_calls_synthesize_sample_realistic(tmp_path, monkeypatch):
    import training.cli as cli_module

    calls = []

    def _fake_realistic(*args, **kwargs):
        calls.append(1)
        return None

    monkeypatch.setattr(cli_module, "synthesize_sample_realistic", _fake_realistic)

    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _write_clip(clips_dir / "bear.mp4")
    profiles_dir = tmp_path / "profiles"
    _write_profiles_dir(profiles_dir)
    out_dir = tmp_path / "out"

    summary = cli_module.run_batch(
        clips_dir, profiles_dir, out_dir, samples_per_combo=1, injection_mode="realistic"
    )

    assert len(calls) == 2  # one call per pattern (general, red)
    assert summary["failed"] == 2  # the fake always returns None -> validation_exhausted


def test_run_batch_raises_when_resuming_into_a_directory_with_a_different_injection_mode(tmp_path):
    clips_dir, profiles_dir = _standard_inputs(tmp_path)
    out_dir = tmp_path / "out"

    run_batch(clips_dir, profiles_dir, out_dir, samples_per_combo=1, seed=0, injection_mode="flat")

    with pytest.raises(ValueError, match="injection_mode"):
        run_batch(clips_dir, profiles_dir, out_dir, samples_per_combo=1, seed=0, injection_mode="realistic")


def test_run_batch_resumes_normally_when_injection_mode_matches(tmp_path):
    clips_dir, profiles_dir = _standard_inputs(tmp_path)
    out_dir = tmp_path / "out"

    run_batch(clips_dir, profiles_dir, out_dir, samples_per_combo=1, seed=0, injection_mode="flat")
    second = run_batch(clips_dir, profiles_dir, out_dir, samples_per_combo=1, seed=1, injection_mode="flat")

    assert second["accepted"] == 0
    assert second["skipped_existing"] == 2


def test_run_batch_overwrite_bypasses_injection_mode_mismatch_check(tmp_path):
    clips_dir, profiles_dir = _standard_inputs(tmp_path)
    out_dir = tmp_path / "out"

    run_batch(clips_dir, profiles_dir, out_dir, samples_per_combo=1, seed=0, injection_mode="flat")
    # overwrite=True regenerates unconditionally, so a mode switch must not raise.
    second = run_batch(
        clips_dir, profiles_dir, out_dir, samples_per_combo=1, seed=0,
        injection_mode="realistic", overwrite=True,
    )

    assert second["skipped_existing"] == 0
    assert second["accepted"] == 2
    meta = json.loads((out_dir / "bear__tiny__general__000" / "meta.json").read_text(encoding="utf-8"))
    assert meta["injection_mode"] == "realistic"


def test_write_sample_records_injection_mode_via_run_batch(tmp_path):
    clips_dir, profiles_dir = _standard_inputs(tmp_path)
    out_dir = tmp_path / "out"

    run_batch(clips_dir, profiles_dir, out_dir, samples_per_combo=1, seed=0, injection_mode="flat")

    meta = json.loads((out_dir / "bear__tiny__general__000" / "meta.json").read_text(encoding="utf-8"))
    assert meta["injection_mode"] == "flat"


def test_run_batch_defaults_to_flat_injection_mode(tmp_path, monkeypatch):
    import training.cli as cli_module

    calls = []

    def _fake_flat(*args, **kwargs):
        calls.append(1)
        return None

    monkeypatch.setattr(cli_module, "synthesize_sample", _fake_flat)

    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _write_clip(clips_dir / "bear.mp4")
    profiles_dir = tmp_path / "profiles"
    _write_profiles_dir(profiles_dir)
    out_dir = tmp_path / "out"

    cli_module.run_batch(clips_dir, profiles_dir, out_dir, samples_per_combo=1)  # no injection_mode passed

    assert len(calls) == 2
