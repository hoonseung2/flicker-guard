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


def _weighted_mean(per_item: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Average `per_item` (B,), weighted per item, normalised by the weight
    total rather than by B.

    This is what makes a weight of 0 remove an item cleanly instead of
    merely shrinking its contribution: normalising by B (a plain weighted
    sum) would still let the remaining items' average drift every time the
    batch's untrusted/trusted mix changes, since the denominator would
    count items that contribute nothing to the numerator. Normalising by
    the weight total instead means a trusted item's result is identical
    whether it arrived alone or alongside any number of zero-weight items.

    If every weight is 0, `total` is 0 and the item-count-weighted quantity
    is undefined -- rather than divide by (approximately) zero and return a
    huge or NaN value, this returns exactly 0, multiplying (not just
    clamping) `per_item` so the zero also carries no gradient. That is the
    right answer for a batch with nothing trustworthy to learn from.
    """
    total = weights.sum()
    if total.item() <= 0.0:
        return (per_item * weights).sum()  # weights are all 0 here, so this is exactly 0
    return (per_item * weights).sum() / total


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
    weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Loss for two consecutive frames, with no clean reference.

    Three terms pull against each other, but not as evenly as that suggests.
    A 400-step Adam optimisation run directly against this loss (see
    `docs/superpowers/specs/2026-08-07-soft-risk-agreement.md`'s companion
    measurement) found:

    - temporal is what actually *moves* the output -- at settling it holds
      35-43% of the gradient magnitude on `out_a`, against 54-63% for
      fidelity and only 1.7-2.3% for risk;
    - risk is a near-threshold finisher, not a driver. At `tau = 0.02`,
      pixels already flagged sit deep in the relaxing sigmoid's saturated
      region, where its gradient is ~0; isolating risk (`lambda_temporal=0,
      lambda_risk=100`) moved a fully-flagged case's flagged area by
      0.0000. `max_area_ratio` is not an attractor the output settles at --
      a global-flash case measured settling at soft area 0.107 (hard 0.000)
      against a limit of 0.25, well clear of it, and an already-compliant
      input (hard 0.0938) still got pushed further, to 0.0811. The
      equilibrium is set by fidelity vs. temporal, not by the limit;
    - risk is nonetheless load-bearing: with `lambda_risk=0`, both
      flickering test cases stayed at their input's flagged area (amplitude
      reduced, but still flagged) -- exactly the "corrects every frame by
      the same proportion, flicker intact" failure this design exists to
      remove. Without risk, nothing in the loss favours crossing the
      threshold over merely approaching it.

    Practically: **`lambda_temporal`, not `lambda_risk`, is the
    over-correction knob.** Turning `lambda_risk` up sharpens the finish
    near the limit; turning `lambda_temporal` up is what flattens more of
    the frame.

    The risk term is why this exists. Grading a frame against a clean
    reference, as the previous objective did, cannot see flicker at all --
    it is satisfied by correcting every frame in the same proportion, which
    is measurably what the trained model learned to do.

    `dx`/`dy` come from `detector.motion.estimate_global_shift` on the INPUT
    pair, not the outputs: the motion is a property of the footage, and
    using the Detector's own estimate keeps the loss scoring the alignment
    the pass/fail rule was computed on.

    `weights` is an optional (B,) per-item weight applied to `temporal`,
    `risk` and `soft_area` only -- `fidelity` doesn't depend on `dx`/`dy`,
    so it has nothing a bad shift estimate could contaminate. `None` (the
    default) weights every item 1, reproducing the plain per-item mean this
    function always used. The caller's motivating case is a rejected
    `estimate_global_shift`: `(dx, dy) = (0, 0)` is both "genuinely did not
    move" and "refused to guess," and passing a trust weight of 0 for the
    second case is what keeps that item from charging its (uncompensated)
    real motion as flicker. The weighting is applied per item BEFORE the
    batch reduction, not as a single scalar multiplied onto the already-
    batch-averaged result -- the latter would still let an untrusted item's
    raw per-item value leak into the average (just diluted), and would also
    shrink a trusted item's contribution depending on how many untrusted
    items happen to share its batch, neither of which is "zeroed."
    """
    # Imported locally: training.soft_risk imports relative_luminance_torch
    # from this module at module level, so a top-level import here would be
    # circular.
    from training.soft_risk import risk_penalty, soft_flagged_area

    # All four terms use the SAME per-item reduction convention: compute one
    # value per batch item first, then reduce across items. That matters
    # once items in a batch have unequal valid-pixel counts (e.g. one item's
    # warp clips more border than another's) -- pooling raw elements across
    # the batch before averaging would silently down-weight the item with
    # fewer valid pixels instead of counting every item equally.
    fidelity_per_item = (
        F.l1_loss(out_a, in_a, reduction="none").mean(dim=(1, 2, 3))
        + F.l1_loss(out_b, in_b, reduction="none").mean(dim=(1, 2, 3))
    )
    fidelity = fidelity_per_item.mean()

    warped_out_a, valid = warp_by_shift(out_a, dx, dy)
    luminance_delta = (
        relative_luminance_torch(warped_out_a) - relative_luminance_torch(out_b)
    ).abs()
    per_item_denominator = valid.sum(dim=(1, 2, 3)).clamp(min=1e-6)
    temporal_per_item = (luminance_delta * valid).sum(dim=(1, 2, 3)) / per_item_denominator

    area_per_item = soft_flagged_area(warped_out_a, out_b, profile, tau=tau, valid_mask=valid)
    risk_per_item = risk_penalty(area_per_item, profile.max_area_ratio, tau_area=tau_area)

    # weights=None reduces to torch.ones, i.e. _weighted_mean(x, ones) ==
    # x.mean() -- the exact expression this function used before `weights`
    # existed, so every existing caller's numbers are unchanged.
    item_weights = weights if weights is not None else torch.ones_like(temporal_per_item)
    temporal = _weighted_mean(temporal_per_item, item_weights)
    risk = _weighted_mean(risk_per_item, item_weights)
    soft_area = _weighted_mean(area_per_item, item_weights)

    return {
        "loss": fidelity + lambda_temporal * temporal + lambda_risk * risk,
        "fidelity": fidelity,
        "temporal": temporal,
        "risk": risk,
        "soft_area": soft_area,
    }
