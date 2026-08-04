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
from prior.compute import FramePrior, compute_prior

_TRAIN_PCT = 80


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
    n_bins = next((p.target_histogram.shape[0] for p in priors if p.target_histogram is not None), 64)
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
    )


def _load_prior_cache(cache_path: Path) -> list[FramePrior]:
    data = np.load(cache_path)
    frame_indices = data["frame_indices"]
    histograms = data["histograms"]
    has_target = data["has_target"]
    masks = data["masks"]
    priors = []
    for i in range(len(frame_indices)):
        target = histograms[i] if has_target[i] else None
        priors.append(FramePrior(
            frame_index=int(frame_indices[i]),
            target_histogram=target,
            mask=masks[i],
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
        self._priors_by_sample: dict[Path, dict[int, FramePrior]] = {}
        self._index: list[tuple[Path, int]] = []

        for sample_dir in sorted(self.data_dir.iterdir()):
            meta_path = sample_dir / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if clip_split(meta["clip_id"]) != split:
                continue

            priors = _load_or_compute_prior(sample_dir, meta["fps"], profile)
            self._priors_by_sample[sample_dir] = {p.frame_index: p for p in priors}

            risky_indices: set[int] = set()
            for segment in meta["segments"]:
                risky_indices.update(range(segment["start_frame"], segment["end_frame"] + 1))

            for prior in priors:
                if prior.frame_index in risky_indices and prior.target_histogram is not None:
                    self._index.append((sample_dir, prior.frame_index))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        sample_dir, frame_index = self._index[idx]
        priors = self._priors_by_sample[sample_dir]
        prior = priors[frame_index]
        n_frames = len(priors)

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
        }
