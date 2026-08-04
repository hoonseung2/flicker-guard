import json

import numpy as np
import pytest
import torch

from detector.profiles import ThresholdProfile
from training.dataset_writer import _write_frames
from training.mitigator_dataset import MitigatorDataset, clip_split

PROFILE = ThresholdProfile(
    name="test", max_flashes_per_second=3, max_area_ratio=0.10,
    general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
    red_saturation_ratio_threshold=0.80,
)


def _write_fake_sample(root, clip_id, n_frames=20, h=8, w=8, fps=10.0, segments=None):
    sample_dir = root / f"{clip_id}__test__general__000"
    frames = [np.full((h, w, 3), 0.5, dtype=np.float32) for _ in range(n_frames)]
    _write_frames(frames, sample_dir / "clean")
    _write_frames(frames, sample_dir / "degraded")
    meta = {
        "clip_id": clip_id,
        "fps": fps,
        "profile": "test",
        "pattern": "general",
        "injection_mode": "flat",
        "segments": segments if segments is not None else [],
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return sample_dir


def test_clip_split_is_deterministic():
    assert clip_split("bear") == clip_split("bear")


def test_clip_split_covers_both_train_and_val_across_many_ids():
    splits = {clip_split(f"clip_{i}") for i in range(50)}
    assert splits == {"train", "val"}


def test_dataset_excludes_frames_outside_segments(tmp_path):
    # frames 0-19 exist, but no segments -> nothing is "risky", so len == 0
    # regardless of what compute_prior finds.
    _write_fake_sample(tmp_path, "clipA", segments=[])
    split = clip_split("clipA")
    dataset = MitigatorDataset(tmp_path, PROFILE, split=split)
    assert len(dataset) == 0


def test_dataset_includes_only_frames_in_segments_with_valid_target(tmp_path):
    # fps=10 -> Detector's trailing window needs ~10 frames before it stops
    # reporting uncertain=True, so target_histogram is None for frames
    # 0-9ish on any clip. Frames 12-15 are past that priming window.
    _write_fake_sample(tmp_path, "clipB", n_frames=20, fps=10.0,
                       segments=[{"start_frame": 12, "end_frame": 15}])
    split = clip_split("clipB")
    dataset = MitigatorDataset(tmp_path, PROFILE, split=split)
    assert len(dataset) == 4


def test_getitem_returns_correct_shapes(tmp_path):
    _write_fake_sample(tmp_path, "clipC", n_frames=20, h=8, w=8, fps=10.0,
                       segments=[{"start_frame": 12, "end_frame": 15}])
    split = clip_split("clipC")
    dataset = MitigatorDataset(tmp_path, PROFILE, split=split)
    example = dataset[0]
    assert example["window"].shape == (9, 8, 8)
    assert example["mask"].shape == (1, 8, 8)
    assert example["histogram"].shape == (64,)
    assert example["clean"].shape == (3, 8, 8)


def test_getitem_duplicates_boundary_frame_at_clip_end(tmp_path):
    # segment reaches the last frame index (19 in a 20-frame clip) -- the
    # window's "next" frame has no real neighbor and must duplicate the
    # center frame instead of indexing out of range.
    _write_fake_sample(tmp_path, "clipD", n_frames=20, h=8, w=8, fps=10.0,
                       segments=[{"start_frame": 12, "end_frame": 19}])
    split = clip_split("clipD")
    dataset = MitigatorDataset(tmp_path, PROFILE, split=split)
    # find the example for frame_index 19 by checking dataset._index directly
    frame_indices = [idx for _, idx in dataset._index]
    assert 19 in frame_indices
    pos = frame_indices.index(19)
    example = dataset[pos]
    center = example["window"][3:6, :, :]
    next_frame = example["window"][6:9, :, :]
    assert torch.equal(center, next_frame)


def test_dataset_only_retains_frame_priors_for_indexed_frames(tmp_path):
    # Only ~10% of a real dataset's frames land in self._index (risky +
    # has-target), so retaining every frame's FramePrior (each carrying a
    # full-resolution mask) for the dataset's lifetime wastes most of the
    # memory it holds. Retained priors must match self._index exactly.
    _write_fake_sample(tmp_path, "clipG", n_frames=20, h=8, w=8, fps=10.0,
                       segments=[{"start_frame": 12, "end_frame": 15}])
    split = clip_split("clipG")
    dataset = MitigatorDataset(tmp_path, PROFILE, split=split)
    sample_dir = tmp_path / "clipG__test__general__000"

    retained_indices = set(dataset._priors_by_sample[sample_dir].keys())
    indexed_indices = {idx for d, idx in dataset._index if d == sample_dir}
    assert retained_indices == indexed_indices
    assert retained_indices < set(range(20))


def test_dataset_tracks_samples_scanned_and_contributing(tmp_path):
    # clipH contributes no examples (no segments); clipI contributes some.
    # Both scenarios must be reflected separately, not just aggregated away
    # into len(dataset).
    _write_fake_sample(tmp_path, "clipH", n_frames=20, fps=10.0, segments=[])
    _write_fake_sample(tmp_path, "clipI", n_frames=20, fps=10.0,
                       segments=[{"start_frame": 12, "end_frame": 15}])

    for split in ("train", "val"):
        dataset = MitigatorDataset(tmp_path, PROFILE, split=split)
        scanned = sum(1 for cid in ("clipH", "clipI") if clip_split(cid) == split)
        contributing = sum(
            1 for cid in ("clipI",) if clip_split(cid) == split
        )
        assert dataset.samples_scanned == scanned
        assert dataset.samples_contributing == contributing


def test_dataset_excludes_risky_frames_with_none_target_histogram(tmp_path):
    # fps=10 -> Detector's trailing window needs ~10 frames before it stops
    # reporting uncertain=True, so target_histogram is None for at least
    # frames 0-9 of any clip. A segment starting at frame 0 straddles that
    # boundary: frames inside the segment but before priming completes must
    # be excluded even though they ARE in a risky segment, while frames
    # after priming (still in the same segment) must be included.
    _write_fake_sample(tmp_path, "clipF", n_frames=20, fps=10.0,
                       segments=[{"start_frame": 0, "end_frame": 15}])
    split = clip_split("clipF")
    dataset = MitigatorDataset(tmp_path, PROFILE, split=split)
    frame_indices = sorted(idx for _, idx in dataset._index)
    assert 0 not in frame_indices  # inside segment, but target_histogram is None this early
    assert 15 in frame_indices     # inside segment, past priming, target_histogram is valid


def test_prior_cache_is_written_and_reused(tmp_path, monkeypatch):
    _write_fake_sample(tmp_path, "clipE", n_frames=20, fps=10.0,
                       segments=[{"start_frame": 12, "end_frame": 15}])
    split = clip_split("clipE")

    MitigatorDataset(tmp_path, PROFILE, split=split)
    sample_dir = tmp_path / "clipE__test__general__000"
    assert (sample_dir / "prior_cache.npz").exists()

    import training.mitigator_dataset as mitigator_dataset_module

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("compute_prior should not be called when a cache file exists")

    monkeypatch.setattr(mitigator_dataset_module, "compute_prior", _fail_if_called)
    MitigatorDataset(tmp_path, PROFILE, split=split)  # must not raise


def test_load_prior_cache_reads_each_key_only_once(tmp_path, monkeypatch):
    # Regression test for NPZ re-decompression bug: _load_prior_cache should
    # materialize each .npz key exactly once before the loop, not re-fetch
    # (and re-decompress) it once per frame.
    from training.mitigator_dataset import (
        _load_prior_cache, _save_prior_cache, FramePrior
    )

    # Create a small synthetic cache with 5 frames to make per-frame bugs obvious
    sample_dir = tmp_path / "test_sample"
    sample_dir.mkdir()
    cache_path = sample_dir / "prior_cache.npz"

    priors = [
        FramePrior(
            frame_index=i,
            target_histogram=np.random.rand(64).astype(np.float32),
            mask=np.ones((8, 8), dtype=bool),
        )
        for i in range(5)
    ]
    _save_prior_cache(cache_path, priors)

    # Monkeypatch np.load to wrap the NpzFile and count __getitem__ calls per key
    import training.mitigator_dataset as mitigator_dataset_module

    call_counts = {}
    original_load = np.load

    class CountingNpzFile:
        def __init__(self, npz_file):
            self._npz = npz_file

        def __getitem__(self, key):
            call_counts[key] = call_counts.get(key, 0) + 1
            return self._npz[key]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self._npz.__exit__(*args)

    def patched_load(path, *args, **kwargs):
        return CountingNpzFile(original_load(path, *args, **kwargs))

    monkeypatch.setattr(mitigator_dataset_module.np, "load", patched_load)

    # Call _load_prior_cache and verify each key was accessed exactly once
    result = _load_prior_cache(cache_path)

    assert len(result) == 5
    for key in ["frame_indices", "histograms", "has_target", "masks"]:
        assert call_counts.get(key) == 1, (
            f"Key '{key}' was accessed {call_counts.get(key, 0)} times, "
            f"expected 1 (indicates per-frame re-fetch bug if > 1)"
        )


def test_dataset_raises_clear_error_when_meta_json_has_no_fps(tmp_path):
    # meta.json written before the fps field existed has no "fps" key --
    # this must fail with a message naming the sample and telling the user
    # how to fix it, not a bare KeyError: 'fps'.
    sample_dir = _write_fake_sample(tmp_path, "clipJ", n_frames=20, fps=10.0,
                                    segments=[{"start_frame": 12, "end_frame": 15}])
    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    del meta["fps"]
    (sample_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    split = clip_split("clipJ")
    with pytest.raises(KeyError, match="fps"):
        MitigatorDataset(tmp_path, PROFILE, split=split)
