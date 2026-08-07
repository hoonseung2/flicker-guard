"""Bridges DatasetSynth's data/synthetic/ output and PriorCalc's
compute_prior into training examples for Mitigator: one example per risky,
prior-conditioned frame, as a (3-frame degraded window, risk mask, target
histogram) -> clean center-frame pair.

Train/val split is a deterministic hash of clip_id, not a stored list --
same philosophy as training.cli's per-sample rng derivation: the split is a
pure function of the input, so it never depends on scan order and never
needs a separate file kept in sync. All samples for a given clip (both
general and red pattern) land on the same side, so no clip's content leaks
across the split.

compute_prior's per-frame illumination histogram is expensive to reproduce
every epoch (it re-runs Detector's full classical pipeline, including
motion estimation, over the whole clip) -- it is computed once per sample
and cached to prior_cache.npz alongside that sample's frames, mirroring
DatasetSynth's own resume-by-checking-what-exists philosophy.
"""
import json
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from detector.profiles import ThresholdProfile
from prior.compute import DEFAULT_N_BINS, FramePrior, compute_prior

_TRAIN_PCT = 80

# Bumped whenever prior_cache.npz's contents change meaning. A cache
# written by an older version is rejected rather than loaded with missing
# or misinterpreted fields -- the failure would otherwise surface as bad
# training rather than an error.
_PRIOR_CACHE_VERSION = 2


def clip_split(clip_id: str) -> str:
    return "train" if zlib.crc32(clip_id.encode("utf-8")) % 100 < _TRAIN_PCT else "val"


def _read_frame(frames_dir: Path, index: int) -> np.ndarray:
    path = frames_dir / f"{index:06d}.png"
    bgr_uint8 = cv2.imread(str(path))
    if bgr_uint8 is None:
        raise OSError(f"failed to read frame: {path}")
    return cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _read_frame_sequence(frames_dir: Path) -> list[np.ndarray]:
    n = len(list(frames_dir.glob("*.png")))
    return [_read_frame(frames_dir, i) for i in range(n)]


def _save_prior_cache(cache_path: Path, priors: list[FramePrior]) -> None:
    n = len(priors)
    n_bins = next((p.target_histogram.shape[0] for p in priors if p.target_histogram is not None), DEFAULT_N_BINS)
    has_target = np.array([p.target_histogram is not None for p in priors], dtype=bool)
    histograms = np.zeros((n, n_bins), dtype=np.float32)
    for i, p in enumerate(priors):
        if p.target_histogram is not None:
            histograms[i] = p.target_histogram
    masks = np.stack([p.mask for p in priors]).astype(bool)
    frame_indices = np.array([p.frame_index for p in priors], dtype=np.int64)
    np.savez_compressed(
        cache_path,
        has_target=has_target,
        histograms=histograms,
        masks=masks,
        frame_indices=frame_indices,
        version=np.array([_PRIOR_CACHE_VERSION], dtype=np.int64),
        required_strengths=np.array([p.required_strength for p in priors], dtype=np.float32),
    )


def _load_prior_cache(cache_path: Path) -> list[FramePrior]:
    with np.load(cache_path) as data:
        # try/except rather than `"version" in data`: NpzFile supports `in`,
        # but this also has to work against the CountingNpzFile stub in
        # test_load_prior_cache_reads_each_key_only_once, which only
        # implements __getitem__ -- `in` on that falls back to the legacy
        # int-indexed iteration protocol and raises, not KeyError.
        try:
            version = int(data["version"][0])
        except KeyError:
            version = 1
        if version != _PRIOR_CACHE_VERSION:
            raise ValueError(
                f"prior cache {cache_path} is version {version}, expected "
                f"{_PRIOR_CACHE_VERSION}. Delete every data/synthetic/*/prior_cache.npz "
                f"and let training regenerate them."
            )
        frame_indices = data["frame_indices"]
        histograms = data["histograms"]
        has_target = data["has_target"]
        masks = data["masks"]
        strengths = data["required_strengths"]
    priors = []
    for i in range(len(frame_indices)):
        target = histograms[i] if has_target[i] else None
        priors.append(FramePrior(
            frame_index=int(frame_indices[i]),
            target_histogram=target,
            mask=masks[i],
            required_strength=float(strengths[i]),
        ))
    return priors


def _load_or_compute_prior(
    sample_dir: Path, fps: float, profile: ThresholdProfile
) -> list[FramePrior]:
    cache_path = sample_dir / "prior_cache.npz"
    if cache_path.exists():
        return _load_prior_cache(cache_path)
    degraded_frames = _read_frame_sequence(sample_dir / "degraded")
    priors = compute_prior(degraded_frames, fps=fps, profile=profile)
    _save_prior_cache(cache_path, priors)
    return priors


class MitigatorDataset(Dataset):
    def __init__(self, data_dir: Path, profile: ThresholdProfile, split: str):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.data_dir = Path(data_dir)
        # Only the FramePrior objects __getitem__ actually reads (the risky,
        # has-target center frames that landed in self._index) are retained
        # here -- each FramePrior carries a full-resolution per-pixel mask,
        # and most of a sample's frames never end up in self._index (see
        # samples_scanned/samples_contributing below), so keeping every
        # frame's FramePrior around for the dataset's lifetime wastes a lot
        # of memory for data that is never read.
        self._priors_by_sample: dict[Path, dict[int, FramePrior]] = {}
        # Frame count per sample, tracked separately from
        # _priors_by_sample so __getitem__'s clip-boundary clamp
        # (max/min against the last frame index) doesn't require keeping
        # every frame's FramePrior alive.
        self._frame_counts: dict[Path, int] = {}
        self._index: list[tuple[Path, int]] = []
        # samples_scanned: every sample_dir in this split with a meta.json.
        # samples_contributing: the subset of those that yielded at least
        # one training example. The gap between the two is normally large
        # (see docs/superpowers/specs/2026-08-03-mitigator-design.md §8) --
        # exposing both here lets a caller (e.g. train_mitigator.main)
        # surface that instead of it being a silent aggregate.
        self.samples_scanned = 0
        self.samples_contributing = 0

        for sample_dir in sorted(self.data_dir.iterdir()):
            meta_path = sample_dir / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if clip_split(meta["clip_id"]) != split:
                continue

            self.samples_scanned += 1
            try:
                fps = meta["fps"]
            except KeyError as exc:
                raise KeyError(
                    f"{sample_dir}: meta.json has no 'fps' field -- this sample "
                    "predates the fps field being written to meta.json. "
                    "Regenerate it with the current training.dataset_writer / "
                    "training.cli."
                ) from exc

            priors = _load_or_compute_prior(sample_dir, fps, profile)
            self._frame_counts[sample_dir] = len(priors)

            risky_indices: set[int] = set()
            for segment in meta["segments"]:
                risky_indices.update(range(segment["start_frame"], segment["end_frame"] + 1))

            sample_priors: dict[int, FramePrior] = {}
            for prior in priors:
                if prior.frame_index in risky_indices and prior.target_histogram is not None:
                    self._index.append((sample_dir, prior.frame_index))
                    sample_priors[prior.frame_index] = prior

            if sample_priors:
                self._priors_by_sample[sample_dir] = sample_priors
                self.samples_contributing += 1

    def __len__(self) -> int:
        return len(self._index)

    def _build_example(self, sample_dir: Path, frame_index: int) -> dict:
        prior = self._priors_by_sample[sample_dir][frame_index]
        n_frames = self._frame_counts[sample_dir]

        prev_idx = max(0, frame_index - 1)
        next_idx = min(n_frames - 1, frame_index + 1)
        degraded_dir = sample_dir / "degraded"
        window = np.concatenate(
            [
                _read_frame(degraded_dir, prev_idx),
                _read_frame(degraded_dir, frame_index),
                _read_frame(degraded_dir, next_idx),
            ],
            axis=-1,
        )
        clean_center = _read_frame(sample_dir / "clean", frame_index)

        return {
            "window": torch.from_numpy(window.transpose(2, 0, 1)).float(),
            "mask": torch.from_numpy(prior.mask[None, :, :].astype(np.float32)),
            "histogram": torch.from_numpy(prior.target_histogram).float(),
            "clean": torch.from_numpy(clean_center.transpose(2, 0, 1)).float(),
            "strength": torch.tensor([prior.required_strength], dtype=torch.float32),
        }

    def __getitem__(self, idx: int) -> dict:
        sample_dir, frame_index = self._index[idx]
        example = self._build_example(sample_dir, frame_index)

        # The temporal consistency term needs two consecutive *restored*
        # frames, which means running the model on a neighbour -- so the
        # neighbour's full input (window, mask, histogram) travels with the
        # example rather than being reconstructed in the training loop.
        #
        # Only risky, has-target frames are retained in _priors_by_sample,
        # so the successor is missing at a segment's last frame. Pairing
        # such a frame with itself makes the predicted and target luminance
        # deltas both zero, contributing nothing to the temporal term --
        # preferable to pairing it with an unrelated frame from elsewhere in
        # the clip, which would charge the model for a difference that isn't
        # flicker. Risk segments are contiguous, so this affects one frame
        # per segment.
        sample_priors = self._priors_by_sample[sample_dir]
        partner_index = frame_index + 1 if (frame_index + 1) in sample_priors else frame_index
        partner = self._build_example(sample_dir, partner_index)

        example["window_partner"] = partner["window"]
        example["mask_partner"] = partner["mask"]
        example["histogram_partner"] = partner["histogram"]
        example["clean_partner"] = partner["clean"]
        example["strength_partner"] = partner["strength"]
        return example
