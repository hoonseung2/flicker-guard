import json

import numpy as np
import pytest

import training.dataset_writer as dataset_writer
from detector.segments import RiskSegment
from training.dataset_writer import (
    sample_exists,
    sample_id,
    sample_injection_mode,
    write_sample,
)
from training.params import InjectionWindow
from training.synth import SynthesizedSample


def _sample():
    window = InjectionWindow(
        start_frame=1, end_frame=10, mask_top=0, mask_left=0,
        mask_height=4, mask_width=4, period_frames=1, ramp_frames=2,
    )
    frames = [np.full((4, 4, 3), 0.5, dtype=np.float32) for _ in range(12)]
    return SynthesizedSample(
        pattern="general", profile_name="kr", window=window,
        clean_frames=frames, degraded_frames=frames,
        segments=[RiskSegment(start_frame=3, end_frame=10)],
    )


def test_sample_id_format():
    assert sample_id("bear", "kr", "general", 0) == "bear__kr__general__000"


def test_write_sample_creates_frame_files_and_metadata(tmp_path):
    sid = write_sample(_sample(), tmp_path, clip_id="bear", index=0, injection_mode="flat", fps=24.0)
    sample_dir = tmp_path / sid
    assert (sample_dir / "clean" / "000000.png").exists()
    assert (sample_dir / "clean" / "000011.png").exists()
    assert (sample_dir / "degraded" / "000000.png").exists()
    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["clip_id"] == "bear"
    assert meta["profile"] == "kr"
    assert meta["pattern"] == "general"
    assert meta["injection_mode"] == "flat"
    assert meta["injected_window"]["start_frame"] == 1
    assert meta["segments"] == [{"start_frame": 3, "end_frame": 10}]


def test_write_sample_records_realistic_injection_mode(tmp_path):
    sid = write_sample(_sample(), tmp_path, clip_id="bear", index=0, injection_mode="realistic", fps=24.0)
    meta = json.loads((tmp_path / sid / "meta.json").read_text(encoding="utf-8"))
    assert meta["injection_mode"] == "realistic"


def test_write_sample_records_fps(tmp_path):
    sid = write_sample(_sample(), tmp_path, clip_id="bear", index=0, injection_mode="flat", fps=24.0)
    meta = json.loads((tmp_path / sid / "meta.json").read_text(encoding="utf-8"))
    assert meta["fps"] == 24.0


def test_sample_exists_true_after_write_false_before(tmp_path):
    sid = sample_id("bear", "kr", "general", 0)
    assert not sample_exists(tmp_path, sid)
    write_sample(_sample(), tmp_path, clip_id="bear", index=0, injection_mode="flat", fps=24.0)
    assert sample_exists(tmp_path, sid)


def test_sample_injection_mode_reads_back_recorded_mode(tmp_path):
    sid = write_sample(_sample(), tmp_path, clip_id="bear", index=0, injection_mode="realistic", fps=24.0)
    assert sample_injection_mode(tmp_path, sid) == "realistic"


def test_sample_injection_mode_defaults_to_flat_when_field_absent(tmp_path):
    # Samples written before this field existed have no "injection_mode" key
    # in meta.json. They were all produced by the (then only) flat path.
    sid = sample_id("bear", "kr", "general", 0)
    sample_dir = tmp_path / sid
    sample_dir.mkdir(parents=True)
    (sample_dir / "meta.json").write_text(json.dumps({"clip_id": "bear"}), encoding="utf-8")
    assert sample_injection_mode(tmp_path, sid) == "flat"


def test_write_sample_raises_and_writes_no_meta_when_a_frame_write_fails(tmp_path, monkeypatch):
    # cv2.imwrite returns False (it does not raise) on disk-full / permission
    # / path-length failures. If meta.json were still written, sample_exists
    # would report the truncated sample as complete and every future resume
    # run would skip it forever.
    monkeypatch.setattr(dataset_writer.cv2, "imwrite", lambda *args, **kwargs: False)

    with pytest.raises(OSError):
        write_sample(_sample(), tmp_path, clip_id="bear", index=0, injection_mode="flat", fps=24.0)

    sid = sample_id("bear", "kr", "general", 0)
    assert not (tmp_path / sid / "meta.json").exists()
    assert not sample_exists(tmp_path, sid)
