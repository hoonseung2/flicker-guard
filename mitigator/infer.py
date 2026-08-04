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
    # compute_prior (detector scoring + motion compensation) has no internal
    # progress hooks and runs single-threaded on CPU regardless of the
    # model's device -- print around it so a slow run doesn't look hung
    # (this bit us in practice: a 20+ minute run produced zero output until
    # completion, which read as a stuck process rather than a slow one).
    print(f"mitigate_segment: computing detection priors for {len(frames)} frames...")
    priors = compute_prior(frames, fps=fps, profile=profile)
    to_correct = sum(1 for prior in priors if prior.target_histogram is not None)
    print(f"mitigate_segment: priors ready, {to_correct} of {len(frames)} frames need correction...")
    was_training = model.training
    model.eval()
    # Run tensors on whichever device the model's weights already live on --
    # mitigate_segment must not silently downgrade a caller's GPU-resident
    # model to CPU (or vice versa) by hardcoding a device here.
    device = next(model.parameters()).device
    n = len(frames)
    output = [frame.copy() for frame in frames]

    corrected_count = 0
    try:
        with torch.no_grad():
            for prior in priors:
                if prior.target_histogram is None:
                    continue  # no conditioning available -- pass through unchanged

                i = prior.frame_index
                prev_idx = max(0, i - 1)
                next_idx = min(n - 1, i + 1)
                window_np = np.concatenate([frames[prev_idx], frames[i], frames[next_idx]], axis=-1)

                window = torch.from_numpy(window_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                mask = torch.from_numpy(prior.mask[None, None, :, :].astype(np.float32)).to(device)
                histogram = torch.from_numpy(prior.target_histogram).float().unsqueeze(0).to(device)

                restored = mitigate_frame(window, mask, histogram, model)
                if not torch.isfinite(restored).all():
                    continue  # NaN/Inf from the model -- keep the original frame, never propagate garbage
                restored_frame = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
                output[i] = np.clip(restored_frame, 0.0, 1.0)
                corrected_count += 1
                print(f"mitigate_segment: corrected frame {i}/{n} ({corrected_count}/{to_correct})...")
    finally:
        model.train(was_training)

    print(f"mitigate_segment: done, corrected {corrected_count} of {n} frames.")
    return output
