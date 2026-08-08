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


def _write_textured_sample(root, clip_id, n_frames=20, h=16, w=16, fps=10.0, segments=None, seed=0):
    # _write_fake_sample's flat frames give phase correlation nothing to
    # lock onto (response ~0.002), so shift_trusted is always False there --
    # never exercising the trusted branch through real computation. Every
    # frame here is the *same* seeded-noise texture repeated across the
    # clip (no injected motion), so consecutive frames correlate against
    # themselves and score a strong response, while still being pixel-
    # identical from frame to frame like _write_fake_sample's flat frames --
    # same "no risk-detector side effects" property, just with structure.
    sample_dir = root / f"{clip_id}__test__general__000"
    rng = np.random.default_rng(seed)
    texture = rng.random((h, w, 3)).astype(np.float32)
    frames = [texture.copy() for _ in range(n_frames)]
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


def test_dataset_includes_only_frames_in_segments(tmp_path):
    # Segment membership alone determines inclusion -- risky_indices covers
    # frames 12-15 inclusive, so exactly those 4 frames land in the index.
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
    # Only ~10% of a real dataset's frames land in self._index (the risky
    # ones), so retaining every frame's FramePrior (each carrying a
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


def test_dataset_excludes_the_detector_warm_up_from_a_segment(tmp_path):
    # This used to assert the opposite -- that segment membership alone
    # gates inclusion, so a segment starting at frame 0 contributed frame 0.
    # That reasoning was about the retired histogram-validity gate and was
    # right on synthetic data, where the injected segment is the truth.
    #
    # It is wrong on real footage. Frame 0 has no predecessor, so the
    # Detector cannot compute a transition and conservatively masks the
    # entire frame; mask persistence then ORs that forward. uncertain_frames
    # equals fps exactly on all 17 screened real clips, and real segments
    # frequently start at frame 0 -- so those frames would teach the model
    # to correct whole frames from a mask that means "unknown", not
    # "flashing".
    _write_fake_sample(tmp_path, "clipF", n_frames=20, fps=10.0,
                       segments=[{"start_frame": 0, "end_frame": 15}])
    split = clip_split("clipF")
    dataset = MitigatorDataset(tmp_path, PROFILE, split=split)
    frame_indices = sorted(idx for _, idx in dataset._index)
    assert frame_indices == [10, 11, 12, 13, 14, 15]  # round(10.0) frames dropped


def test_dataset_keeps_a_segment_that_starts_after_the_warm_up_intact(tmp_path):
    # The exclusion is a floor on the frame index, not a blanket trim: a
    # segment wholly past the warm-up must be untouched.
    _write_fake_sample(tmp_path, "clipG", n_frames=20, fps=10.0,
                       segments=[{"start_frame": 12, "end_frame": 15}])
    dataset = MitigatorDataset(tmp_path, PROFILE, split=clip_split("clipG"))
    assert sorted(idx for _, idx in dataset._index) == [12, 13, 14, 15]


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
    for key in ["frame_indices", "masks", "required_strengths"]:
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


def test_getitem_returns_a_partner_frame_for_the_temporal_loss(tmp_path):
    # The self-supervised temporal term compares two consecutive restored
    # frames, so one example has to carry everything needed to run the
    # model on its neighbour as well.
    _write_fake_sample(tmp_path, "clipT", n_frames=20, h=8, w=8, fps=10.0,
                       segments=[{"start_frame": 12, "end_frame": 15}])
    dataset = MitigatorDataset(tmp_path, PROFILE, split=clip_split("clipT"))
    example = dataset[0]
    assert example["window_partner"].shape == (9, 8, 8)
    assert example["mask_partner"].shape == (1, 8, 8)


def test_partner_is_the_next_frame_when_it_is_also_indexed(tmp_path):
    _write_fake_sample(tmp_path, "clipU", n_frames=20, h=8, w=8, fps=10.0,
                       segments=[{"start_frame": 12, "end_frame": 15}])
    dataset = MitigatorDataset(tmp_path, PROFILE, split=clip_split("clipU"))
    frame_indices = [idx for _, idx in dataset._index]
    pos = frame_indices.index(min(frame_indices))
    example = dataset[pos]
    # The partner's window is centred one frame later, so the partner's
    # centre must equal this example's "next" slice.
    assert torch.equal(example["window"][6:9], example["window_partner"][3:6])


def test_partner_falls_back_to_the_frame_itself_at_a_segment_end(tmp_path):
    # The last frame of a segment has no indexed successor. Pairing it with
    # itself makes both the predicted and the target luminance delta zero,
    # so it contributes nothing to the temporal term rather than inventing
    # a partner from an unrelated part of the clip.
    _write_fake_sample(tmp_path, "clipV", n_frames=20, h=8, w=8, fps=10.0,
                       segments=[{"start_frame": 12, "end_frame": 15}])
    dataset = MitigatorDataset(tmp_path, PROFILE, split=clip_split("clipV"))
    frame_indices = [idx for _, idx in dataset._index]
    pos = frame_indices.index(max(frame_indices))
    example = dataset[pos]
    assert torch.equal(example["window"], example["window_partner"])
    assert torch.equal(example["mask"], example["mask_partner"])


def test_getitem_carries_the_required_strength(tmp_path):
    _write_fake_sample(tmp_path, "clipS", n_frames=20, h=8, w=8, fps=10.0,
                       segments=[{"start_frame": 12, "end_frame": 15}])
    dataset = MitigatorDataset(tmp_path, PROFILE, split=clip_split("clipS"))
    example = dataset[0]
    assert example["strength"].shape == (1,)
    assert 0.0 <= float(example["strength"]) <= 1.0
    assert example["strength_partner"].shape == (1,)


def test_getitem_carries_the_shift_to_its_partner(tmp_path):
    _write_fake_sample(tmp_path, "clipM", n_frames=20, h=16, w=16, fps=10.0,
                       segments=[{"start_frame": 12, "end_frame": 15}])
    dataset = MitigatorDataset(tmp_path, PROFILE, split=clip_split("clipM"))
    example = dataset[0]
    assert example["shift"].shape == (2,)
    assert example["shift_trusted"].shape == (1,)
    assert float(example["shift_trusted"]) in (0.0, 1.0)


def test_a_self_paired_frame_reports_no_shift(tmp_path):
    # At a segment's last frame the partner is the frame itself, so there is
    # no motion between them by construction. A nonzero shift there would
    # warp a frame against itself and charge the model for the difference.
    _write_fake_sample(tmp_path, "clipN", n_frames=20, h=16, w=16, fps=10.0,
                       segments=[{"start_frame": 12, "end_frame": 15}])
    dataset = MitigatorDataset(tmp_path, PROFILE, split=clip_split("clipN"))
    frame_indices = [idx for _, idx in dataset._index]
    example = dataset[frame_indices.index(max(frame_indices))]
    assert float(example["shift"][0]) == 0.0
    assert float(example["shift"][1]) == 0.0
    # Trusted precisely because it is not a guess -- there is nothing to
    # reject when there is no phase correlation call at all.
    assert float(example["shift_trusted"]) == 1.0


def test_an_untrusted_shift_is_flagged_rather_than_silently_zeroed(tmp_path, monkeypatch):
    # estimate_global_shift returns (0.0, 0.0) both when the frames really
    # did not move and when it refused to guess. The loss cannot tell those
    # apart, and on a rejected pan it would charge real camera motion as
    # flicker -- with none of the Detector's three smoothing stages to
    # absorb it. Measured: 33.6% of pairs rejected on one real clip.
    import training.mitigator_dataset as dataset_module

    monkeypatch.setattr(
        dataset_module, "estimate_global_shift_checked", lambda a, b: (0.0, 0.0, False)
    )

    _write_fake_sample(tmp_path, "clipU", n_frames=20, h=16, w=16, fps=10.0,
                       segments=[{"start_frame": 12, "end_frame": 15}])
    dataset = MitigatorDataset(tmp_path, PROFILE, split=clip_split("clipU"))
    assert float(dataset[0]["shift_trusted"]) == 0.0


def test_a_textured_pair_is_trusted_by_real_phase_correlation(tmp_path):
    # _write_fake_sample's flat frames score too low (~0.002) to ever reach
    # shift_trusted == 1.0 through real computation -- every existing test
    # touching "shift_trusted" either hits the self-paired shortcut or a
    # monkeypatch. This drives an actual textured pair through
    # estimate_global_shift_checked so the trusted branch is exercised for
    # real: identical frames with real structure correlate with a strong
    # peak (response near 1.0), well above the 0.30 guard.
    _write_textured_sample(tmp_path, "clipW", n_frames=20, h=16, w=16, fps=10.0,
                           segments=[{"start_frame": 12, "end_frame": 15}])
    dataset = MitigatorDataset(tmp_path, PROFILE, split=clip_split("clipW"))
    frame_indices = [idx for _, idx in dataset._index]
    # Use the segment's first indexed frame so the partner is a real
    # neighbour, not the self-paired shortcut at the segment's last frame.
    pos = frame_indices.index(min(frame_indices))
    example = dataset[pos]
    assert float(example["shift_trusted"]) == 1.0


def test_a_stale_prior_cache_is_rejected_rather_than_misread(tmp_path):
    # The cache stores derived arrays with no self-description. A cache
    # written before required_strength existed would otherwise load with a
    # missing field and fail somewhere far from the cause.
    import numpy as np
    from training.mitigator_dataset import _load_prior_cache

    stale = tmp_path / "prior_cache.npz"
    np.savez_compressed(
        stale,
        masks=np.ones((1, 4, 4), dtype=bool),
        frame_indices=np.array([0], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="prior cache"):
        _load_prior_cache(stale)


def _write_real_sample(root, clip_id, n_frames=20, h=8, w=8, fps=10.0):
    """A sample in the shape scripts.ingest_real_clips produces: no 'segments'
    key and no clean/ directory, because a real clip has neither."""
    sample_dir = root / clip_id
    frames = [np.full((h, w, 3), 0.5, dtype=np.float32) for _ in range(n_frames)]
    _write_frames(frames, sample_dir / "degraded")
    meta = {"clip_id": clip_id, "source": f"{clip_id}.mp4", "fps": fps,
            "frames": n_frames, "shape": [h, w]}
    (sample_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return sample_dir


def _fake_priors(n_frames, risky, h=8, w=8):
    from prior.compute import FramePrior
    return [
        FramePrior(frame_index=i, mask=np.ones((h, w), dtype=bool),
                   required_strength=0.5 if i in risky else 0.0)
        for i in range(n_frames)
    ]


def test_dataset_derives_risky_frames_from_priors_when_meta_has_no_segments(tmp_path, monkeypatch):
    # Ingested real clips carry no "segments": producing them would rerun the
    # detection compute_prior already runs, and _assign_segment_strengths
    # sets required_strength above 0 on exactly a RiskSegment's frames.
    import training.mitigator_dataset as dataset_module

    _write_real_sample(tmp_path, "realclip", n_frames=20, fps=1.0)
    monkeypatch.setattr(dataset_module, "_load_or_compute_prior",
                        lambda *a, **k: _fake_priors(20, risky={12, 13, 14}))

    dataset = MitigatorDataset(tmp_path, PROFILE, split="train",
                               split_map={"realclip": "train"})

    assert sorted(idx for _, idx in dataset._index) == [12, 13, 14]


def test_warm_up_exclusion_applies_to_prior_derived_risky_frames_too(tmp_path, monkeypatch):
    import training.mitigator_dataset as dataset_module

    _write_real_sample(tmp_path, "realclip", n_frames=20, fps=5.0)
    monkeypatch.setattr(dataset_module, "_load_or_compute_prior",
                        lambda *a, **k: _fake_priors(20, risky=set(range(20))))

    dataset = MitigatorDataset(tmp_path, PROFILE, split="train",
                               split_map={"realclip": "train"})

    assert sorted(idx for _, idx in dataset._index) == list(range(5, 20))


def test_split_map_overrides_the_hash(tmp_path, monkeypatch):
    import training.mitigator_dataset as dataset_module

    for clip_id in ("alpha", "beta"):
        _write_real_sample(tmp_path, clip_id, n_frames=20, fps=1.0)
    monkeypatch.setattr(dataset_module, "_load_or_compute_prior",
                        lambda *a, **k: _fake_priors(20, risky={12, 13}))
    split_map = {"alpha": "train", "beta": "val"}

    train = MitigatorDataset(tmp_path, PROFILE, split="train", split_map=split_map)
    val = MitigatorDataset(tmp_path, PROFILE, split="val", split_map=split_map)

    assert train.samples_scanned == 1
    assert val.samples_scanned == 1


def test_split_map_errors_when_a_clip_on_disk_is_absent_from_it(tmp_path, monkeypatch):
    # The hash split needed no file kept in sync; an explicit map does. A
    # clip missing from the map lands in neither split and is never trained
    # on, which looks like a smaller corpus rather than an error.
    import training.mitigator_dataset as dataset_module

    _write_real_sample(tmp_path, "alpha", n_frames=20, fps=1.0)
    monkeypatch.setattr(dataset_module, "_load_or_compute_prior",
                        lambda *a, **k: _fake_priors(20, risky={12}))

    with pytest.raises(ValueError, match="missing from the split map"):
        MitigatorDataset(tmp_path, PROFILE, split="train", split_map={"beta": "train"})


def test_split_map_errors_when_it_names_a_clip_that_is_not_on_disk(tmp_path, monkeypatch):
    import training.mitigator_dataset as dataset_module

    _write_real_sample(tmp_path, "alpha", n_frames=20, fps=1.0)
    monkeypatch.setattr(dataset_module, "_load_or_compute_prior",
                        lambda *a, **k: _fake_priors(20, risky={12}))

    with pytest.raises(ValueError, match="not found on disk"):
        MitigatorDataset(tmp_path, PROFILE, split="train",
                         split_map={"alpha": "train", "ghost": "val"})


def test_split_map_is_optional_so_the_synthetic_path_is_unchanged(tmp_path):
    # Phase 1 has to stay reproducible: with no map the clip_id hash decides,
    # exactly as before.
    _write_fake_sample(tmp_path, "clipH", n_frames=20, fps=10.0,
                       segments=[{"start_frame": 12, "end_frame": 15}])
    dataset = MitigatorDataset(tmp_path, PROFILE, split=clip_split("clipH"))
    assert dataset.samples_scanned == 1
