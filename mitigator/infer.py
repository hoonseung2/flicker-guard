"""Evaluation-only inference wrapper: runs a Mitigator model over a
sequence of frames, sliding a 3-frame window and conditioning each frame on
PriorCalc's target histogram and risk mask. Not wired into any runtime
pipeline yet -- Verifier/Fallback/BufferManager integration is a separate,
later roadmap step (see docs/superpowers/specs/2026-08-03-mitigator-design.md
section 2). This exists so a trained checkpoint's output can be inspected
and measured (PSNR/SSIM vs ground truth) during and after training.
"""
import numpy as np
import torch

from detector.profiles import ThresholdProfile
from mitigator.arch import MitigatorNet, mitigate_frame
from prior.compute import compute_prior


def mitigate_segment(
    frames: list[np.ndarray],
    fps: float,
    profile: ThresholdProfile,
    model: MitigatorNet,
) -> list[np.ndarray]:
    priors = compute_prior(frames, fps=fps, profile=profile)
    model.eval()
    n = len(frames)
    output = [frame.copy() for frame in frames]

    with torch.no_grad():
        for prior in priors:
            if prior.target_histogram is None:
                continue  # no conditioning available -- pass through unchanged

            i = prior.frame_index
            prev_idx = max(0, i - 1)
            next_idx = min(n - 1, i + 1)
            window_np = np.concatenate([frames[prev_idx], frames[i], frames[next_idx]], axis=-1)

            window = torch.from_numpy(window_np.transpose(2, 0, 1)).float().unsqueeze(0)
            mask = torch.from_numpy(prior.mask[None, None, :, :].astype(np.float32))
            histogram = torch.from_numpy(prior.target_histogram).float().unsqueeze(0)

            restored = mitigate_frame(window, mask, histogram, model)
            if not torch.isfinite(restored).all():
                continue  # NaN/Inf from the model -- keep the original frame, never propagate garbage
            output[i] = restored.squeeze(0).permute(1, 2, 0).numpy()

    return output
