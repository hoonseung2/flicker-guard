"""Differentiable SSIM/PSNR and the combined L1+SSIM training loss for
Mitigator. Hand-implemented rather than pulled from torchmetrics: SSIM must
be differentiable to use as a training loss (not just an eval-time metric),
and the standard windowed formulation (Wang et al. 2004) is short enough
that adding a new dependency isn't worth it.
"""
import torch
import torch.nn.functional as F

from detector.profiles import ThresholdProfile

_C1 = 0.01 ** 2
_C2 = 0.03 ** 2

# Kept in step with detector.luminance -- see relative_luminance_torch.
_SRGB_A = 0.055
_R_COEFF, _G_COEFF, _B_COEFF = 0.2126, 0.7152, 0.0722


def _gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _ssim_window(window_size: int, channels: int) -> torch.Tensor:
    g1d = _gaussian_kernel(window_size, sigma=1.5)
    g2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)
    return g2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Mean SSIM over the batch. Inputs are (B, C, H, W) float tensors,
    values expected in [0, 1]."""
    channels = img1.shape[1]
    window = _ssim_window(window_size, channels).to(img1.device, img1.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + _C1) * (2 * sigma12 + _C2)) / (
        (mu1_sq + mu2_sq + _C1) * (sigma1_sq + sigma2_sq + _C2)
    )
    return ssim_map.mean()


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    return 10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))


def l1_ssim_loss(pred: torch.Tensor, target: torch.Tensor, lambda_ssim: float = 0.5) -> torch.Tensor:
    l1 = F.l1_loss(pred, target)
    return l1 + lambda_ssim * (1.0 - ssim(pred, target))


def relative_luminance_torch(frames: torch.Tensor) -> torch.Tensor:
    """Differentiable relative luminance for (B, 3, H, W) sRGB tensors.

    Mirrors detector.luminance.relative_luminance exactly. That match is the
    point: the temporal loss below is only meaningful if it measures the
    quantity the Detector actually thresholds on, so this is a torch
    transcription of that function rather than a cheaper approximation.
    """
    # `restored = center + delta` is unclamped, so a prediction can go
    # negative -- and a negative base under the fractional exponent below
    # yields NaN, which backward() would spread through every weight. Both
    # branches of the where() are evaluated, so the floor has to be applied
    # to the input, not to the selected branch.
    safe = frames.clamp(min=0.0)
    linear = torch.where(
        safe <= 0.04045,
        safe / 12.92,
        ((safe + _SRGB_A) / (1 + _SRGB_A)) ** 2.4,
    )
    return (
        _R_COEFF * linear[:, 0:1]
        + _G_COEFF * linear[:, 1:2]
        + _B_COEFF * linear[:, 2:3]
    )


def temporal_consistency_loss(
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
) -> torch.Tensor:
    """Penalise per-pixel luminance change between two consecutive frames
    that the ground-truth pair does not have.

    l1_ssim_loss scores each frame on its own, so a model that corrects
    every frame by the same proportion scores well while leaving the
    original flicker pattern intact at reduced amplitude -- measured on a
    real clip as a 60% smaller brightness jump but only 12% fewer
    transitions, which a viewer still reads as flickering at the same rate.

    Comparing the prediction's frame-to-frame delta against the target's
    (rather than driving the delta to zero) is what keeps legitimate motion:
    a camera move or a cut appears in both, cancels, and costs nothing. Only
    change with no counterpart in the clean pair is charged for.
    """
    delta_pred = relative_luminance_torch(pred_b) - relative_luminance_torch(pred_a)
    delta_target = relative_luminance_torch(target_b) - relative_luminance_torch(target_a)
    return (delta_pred - delta_target).abs().mean()


def warp_by_shift(
    frames: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate `frames` by a per-item global shift, differentiably.

    This is the torch counterpart of `detector.motion.compensate_shift`, and
    it must stay one: the temporal loss compares two frames after cancelling
    camera motion, and if it cancels a different motion than the Detector
    did, it scores an alignment the pass/fail rule never saw.

    `compensate_shift` fills the vacated edge with BORDER_REPLICATE, which
    invents correspondences that do not exist. Here the edge is zero-filled
    and reported through the returned validity mask instead, so a caller can
    exclude it rather than average fabricated pixels into a score.

    Args:
        frames: (B, C, H, W).
        dx, dy: (B,) shifts in pixels, same sign convention as
            `compensate_shift` (positive dx moves content right).

    Returns:
        (warped, valid) where `valid` is (B, 1, H, W), 1 where the sample
        came from inside the source and 0 where it did not.
    """
    batch, _channels, height, width = frames.shape
    device, dtype = frames.device, frames.dtype

    # affine_grid works in normalised [-1, 1] coordinates, and its theta maps
    # OUTPUT coordinates to INPUT ones -- so a shift of +dx pixels of content
    # to the right means sampling dx pixels to the LEFT, hence the negation.
    theta = torch.zeros(batch, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = -2.0 * dx.to(device=device, dtype=dtype) / max(width - 1, 1)
    theta[:, 1, 2] = -2.0 * dy.to(device=device, dtype=dtype) / max(height - 1, 1)

    grid = F.affine_grid(theta, (batch, 1, height, width), align_corners=True)
    warped = F.grid_sample(frames, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    ones = torch.ones(batch, 1, height, width, device=device, dtype=dtype)
    valid = F.grid_sample(ones, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return warped, valid


def self_supervised_loss(
    out_a: torch.Tensor,
    out_b: torch.Tensor,
    in_a: torch.Tensor,
    in_b: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
    profile: ThresholdProfile,
    lambda_temporal: float = 1.0,
    lambda_risk: float = 1.0,
    tau: float = 0.02,
    tau_area: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Loss for two consecutive frames, with no clean reference.

    Three terms pull against each other:

    - fidelity keeps the output on its input, so doing nothing is the
      default and every change has to be paid for;
    - temporal drives consecutive outputs together after cancelling camera
      motion, which is what actually removes flicker;
    - risk charges only for exceeding the Detector's own area limit, so the
      correction stops as soon as the clip is compliant instead of
      flattening everything a global weight happens to reach.

    The risk term is why this exists. Grading a frame against a clean
    reference, as the previous objective did, cannot see flicker at all --
    it is satisfied by correcting every frame in the same proportion, which
    is measurably what the trained model learned to do.

    `dx`/`dy` come from `detector.motion.estimate_global_shift` on the INPUT
    pair, not the outputs: the motion is a property of the footage, and
    using the Detector's own estimate keeps the loss scoring the alignment
    the pass/fail rule was computed on.
    """
    # Imported locally: training.soft_risk imports relative_luminance_torch
    # from this module at module level, so a top-level import here would be
    # circular.
    from training.soft_risk import risk_penalty, soft_flagged_area

    fidelity = F.l1_loss(out_a, in_a) + F.l1_loss(out_b, in_b)

    warped_out_a, valid = warp_by_shift(out_a, dx, dy)
    luminance_delta = (
        relative_luminance_torch(warped_out_a) - relative_luminance_torch(out_b)
    ).abs()
    denominator = valid.sum().clamp(min=1e-6)
    temporal = (luminance_delta * valid).sum() / denominator

    area = soft_flagged_area(warped_out_a, out_b, profile, tau=tau, valid_mask=valid)
    risk = risk_penalty(area, profile.max_area_ratio, tau_area=tau_area).mean()

    return {
        "loss": fidelity + lambda_temporal * temporal + lambda_risk * risk,
        "fidelity": fidelity,
        "temporal": temporal,
        "risk": risk,
        "soft_area": area.mean(),
    }
