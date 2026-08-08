"""Evaluation-only inference wrapper: runs a Mitigator model over a
sequence of frames, sliding a 3-frame window and conditioning each frame on
PriorCalc's required strength and risk mask. Not wired into any runtime
pipeline yet -- Verifier/Fallback/BufferManager integration is a separate,
later roadmap step (see docs/superpowers/specs/2026-08-03-mitigator-design.md
section 2). This exists so a trained checkpoint's output can be inspected
and measured (PSNR/SSIM vs ground truth) during and after training.
"""
import cv2
import numpy as np
import torch

from detector.profiles import ThresholdProfile
from mitigator.arch import MitigatorNet, mitigate_frame
from prior.compute import compute_prior

# The bottleneck's self-attention is global over all H*W spatial positions,
# so its cost is quadratic in pixel count. Training only ever exercised it
# on <=512x512 crops (training.train_mitigator's --patch-size); feeding a
# real video's full native resolution (e.g. 720x1280, ~3.5x the pixels of a
# 512x512 crop, ~12x the attention-matrix size) blew past a T4's VRAM and
# crashed with no Python traceback. Cap the longer side at the training
# patch size so inference never asks the attention layer to run outside the
# regime it was trained and validated on.
_MAX_PROCESSING_DIM = 512


def mitigate_segment(
    frames: list[np.ndarray],
    fps: float,
    profile: ThresholdProfile,
    model: MitigatorNet,
    preserve_hue: bool = True,
) -> list[np.ndarray]:
    # compute_prior (detector scoring + motion compensation) has no internal
    # progress hooks and runs single-threaded on CPU regardless of the
    # model's device -- print around it so a slow run doesn't look hung
    # (this bit us in practice: a 20+ minute run produced zero output until
    # completion, which read as a stuck process rather than a slow one).
    print(f"mitigate_segment: computing detection priors for {len(frames)} frames...", flush=True)
    priors = compute_prior(frames, fps=fps, profile=profile)
    # Strength conditioning is always available (required_strength defaults
    # to 0.0), unlike the old histogram, which was None until enough
    # non-masked reference pixels had accumulated -- so "is there
    # conditioning" is no longer the right question. A frame needs
    # correction when it sits inside an actual risk segment: required_strength
    # is only ever set above 0.0 by _assign_segment_strengths for frames of a
    # detected RiskSegment, so it (together with a non-empty mask) is exactly
    # that signal. mask alone is not enough -- Detector conservatively masks
    # the whole frame during its uncertain warm-up window even when nothing
    # risky has actually been found, and required_strength stays 0.0 for
    # those frames. mitigate_frame's hard blend would return the input
    # byte-for-byte wherever mask == 0 regardless of strength, so this gate
    # is purely about skipping wasted forward passes, not pixel correctness.
    to_correct = sum(1 for prior in priors if prior.required_strength > 0.0 and prior.mask.any())
    print(f"mitigate_segment: priors ready, {to_correct} of {len(frames)} frames need correction...", flush=True)
    was_training = model.training
    model.eval()
    # Run tensors on whichever device the model's weights already live on --
    # mitigate_segment must not silently downgrade a caller's GPU-resident
    # model to CPU (or vice versa) by hardcoding a device here.
    device = next(model.parameters()).device
    n = len(frames)
    # A shallow list copy is enough: nothing in this function ever mutates a
    # frame array in place (concatenate/resize/assignment all produce new
    # arrays), only `output[i]` gets reassigned for corrected frames. Deep-
    # copying every frame here used to double the resident memory of the
    # whole clip (~5GB extra on a 507-frame 720x1280 clip) for no reason --
    # on a RAM-constrained host (e.g. free-tier Colab) that was enough to
    # get the process OOM-killed with no traceback, which looked identical
    # to a hang.
    output = list(frames)

    corrected_count = 0
    try:
        with torch.no_grad():
            for prior in priors:
                if not (prior.required_strength > 0.0 and prior.mask.any()):
                    continue  # not in a risk segment -- nothing to correct, would be a wasted forward pass

                i = prior.frame_index
                prev_idx = max(0, i - 1)
                next_idx = min(n - 1, i + 1)

                orig_h, orig_w = frames[i].shape[:2]
                scale = min(1.0, _MAX_PROCESSING_DIM / max(orig_h, orig_w))
                if scale < 1.0:
                    proc_w, proc_h = round(orig_w * scale), round(orig_h * scale)
                    # cv2.resize rejects >4 channels, so resize each frame
                    # individually (3 channels each) before concatenating
                    # into the 9-channel window the model expects.
                    window_proc = np.concatenate(
                        [
                            cv2.resize(frames[prev_idx], (proc_w, proc_h), interpolation=cv2.INTER_AREA),
                            cv2.resize(frames[i], (proc_w, proc_h), interpolation=cv2.INTER_AREA),
                            cv2.resize(frames[next_idx], (proc_w, proc_h), interpolation=cv2.INTER_AREA),
                        ],
                        axis=-1,
                    )
                    mask_proc = cv2.resize(
                        prior.mask.astype(np.float32), (proc_w, proc_h), interpolation=cv2.INTER_AREA
                    )
                else:
                    window_proc = np.concatenate([frames[prev_idx], frames[i], frames[next_idx]], axis=-1)
                    mask_proc = prior.mask.astype(np.float32)

                window = torch.from_numpy(window_proc.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                mask = torch.from_numpy(mask_proc[None, None, :, :]).to(device)
                strength = torch.tensor([[prior.required_strength]], device=device)

                restored = mitigate_frame(window, mask, strength, model,
                                          preserve_hue=preserve_hue)
                if not torch.isfinite(restored).all():
                    continue  # NaN/Inf from the model -- keep the original frame, never propagate garbage
                restored_frame = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()

                if scale < 1.0:
                    restored_frame = cv2.resize(restored_frame, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    # Re-blend against the ORIGINAL full-resolution mask/center
                    # rather than trusting the upsampled result at the mask
                    # boundary -- interpolation can smear corrected pixels a
                    # little outside the mask, and pixels outside the mask
                    # must remain exactly the original input, unconditionally.
                    full_mask = prior.mask.astype(np.float32)[:, :, None]
                    restored_frame = full_mask * restored_frame + (1 - full_mask) * frames[i]

                output[i] = np.clip(restored_frame, 0.0, 1.0)
                corrected_count += 1
                print(f"mitigate_segment: corrected frame {i}/{n} ({corrected_count}/{to_correct})...", flush=True)
    finally:
        model.train(was_training)

    print(f"mitigate_segment: done, corrected {corrected_count} of {n} frames.", flush=True)
    return output
