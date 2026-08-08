import json

import cv2
import numpy as np
import pytest

from scripts.ingest_real_clips import clip_id_for, ingest_clip, main


def test_clip_id_is_ascii_and_deterministic():
    name = "FE!N in London was epic \U0001f92f\U0001f4a5 #travisscott.mp4"
    first = clip_id_for(name)
    assert first == clip_id_for(name)
    assert first.isascii()


def test_clip_id_distinguishes_names_that_slug_identically():
    # Both collapse to the same slug -- only the hash keeps them apart. A
    # collision would silently merge two clips into one sample directory.
    a = clip_id_for("녹음 2026-08-08 000241.mp4")
    b = clip_id_for("@@@@@ 2026-08-08 000241.mp4")
    assert a.rsplit("-", 1)[0] == b.rsplit("-", 1)[0]
    assert a != b


def test_clip_id_keeps_readable_stem_for_plain_names():
    assert clip_id_for("11832-233049403_medium.mp4").startswith("11832-233049403_medium-")


def _write_video(path, frames, fps=30.0):
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def _synthetic_clip(tmp_path, n_frames, height=64, width=128, fps=30.0, name="src.mp4"):
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (height, width, 3), dtype=np.uint8) for _ in range(n_frames)]
    path = tmp_path / name
    _write_video(path, frames, fps)
    return path


def test_ingest_clip_writes_frames_and_meta(tmp_path):
    source = _synthetic_clip(tmp_path, n_frames=40)
    out = tmp_path / "out"

    meta = ingest_clip(source, out, max_dim=64)

    sample_dir = out / meta["clip_id"]
    written = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    assert written == meta
    assert written["source"] == "src.mp4"
    assert written["fps"] > 0
    assert written["frames"] == len(list((sample_dir / "degraded").glob("*.png")))
    assert max(written["shape"]) == 64


def test_ingest_clip_frames_are_zero_padded_and_contiguous(tmp_path):
    # MitigatorDataset._read_frame builds paths as f"{index:06d}.png" and
    # _read_frame_sequence assumes indices 0..n-1 with no gaps.
    source = _synthetic_clip(tmp_path, n_frames=40)
    out = tmp_path / "out"
    meta = ingest_clip(source, out, max_dim=64)
    degraded = out / meta["clip_id"] / "degraded"
    for i in range(meta["frames"]):
        assert (degraded / f"{i:06d}.png").exists()


def test_ingest_clip_round_trips_pixels_losslessly(tmp_path):
    # The dataset reads these PNGs back with cv2.imread and converts BGR to
    # RGB float32. Writing the wrong channel order would train the model on
    # swapped colours, which the red-flash path would read as a different
    # scene entirely -- and nothing downstream would raise.
    source = _synthetic_clip(tmp_path, n_frames=40)
    out = tmp_path / "out"
    meta = ingest_clip(source, out, max_dim=64)

    from scripts.screen_clean_clips import read_frames_lenient
    frames, _fps, _note = read_frames_lenient(source, 64)
    written = cv2.imread(str(out / meta["clip_id"] / "degraded" / "000000.png"))
    written_rgb = cv2.cvtColor(written, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    assert np.abs(written_rgb - frames[0]).max() <= 1.0 / 255.0


def test_ingest_clip_rejects_a_clip_too_short_to_survive_warmup(tmp_path):
    # The Detector's warm-up is one second and those frames are excluded as
    # training centres, so a clip this short would contribute nothing --
    # far away from here, as a silent zero rather than an error.
    source = _synthetic_clip(tmp_path, n_frames=12, fps=30.0)
    with pytest.raises(ValueError, match="too short"):
        ingest_clip(source, tmp_path / "out", max_dim=64)


def test_ingest_clip_skips_an_existing_sample_unless_overwritten(tmp_path):
    source = _synthetic_clip(tmp_path, n_frames=40)
    out = tmp_path / "out"
    meta = ingest_clip(source, out, max_dim=64)
    marker = out / meta["clip_id"] / "degraded" / "000000.png"
    marker.write_bytes(b"")

    ingest_clip(source, out, max_dim=64)
    assert marker.read_bytes() == b""

    ingest_clip(source, out, max_dim=64, overwrite=True)
    assert marker.read_bytes() != b""


def test_main_ingests_every_clip_except_excluded_ones(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    for name in ("keep_a.mp4", "keep_b.mp4", "drop_me.mp4"):
        _synthetic_clip(tmp_path, n_frames=40, name=name).replace(clips / name)
    out = tmp_path / "out"

    exit_code = main([
        "--clips-dir", str(clips), "--output", str(out),
        "--max-dim", "64", "--exclude", "drop_me.mp4",
    ])

    assert exit_code == 0
    ingested = {
        json.loads((d / "meta.json").read_text(encoding="utf-8"))["source"]
        for d in out.iterdir() if d.is_dir()
    }
    assert ingested == {"keep_a.mp4", "keep_b.mp4"}


def test_main_errors_when_an_exclude_matches_nothing(tmp_path):
    # A mistyped exclusion silently trains on a clip meant to be dropped --
    # including one dropped for evaluation-set contamination.
    clips = tmp_path / "clips"
    clips.mkdir()
    _synthetic_clip(tmp_path, n_frames=40, name="keep.mp4").replace(clips / "keep.mp4")

    with pytest.raises(ValueError, match="matched no file"):
        main([
            "--clips-dir", str(clips), "--output", str(tmp_path / "out"),
            "--max-dim", "64", "--exclude", "typo.mp4",
        ])
