"""A differentiable stand-in for the Detector's risk criterion.

`detector.segments._is_risky` fires on hard thresholds, which have zero
gradient, so a loss cannot be built from it directly. This module relaxes
each threshold through a sigmoid while keeping the SAME threshold values and
the SAME combining structure as `detector.flash` -- the general-flash test
ANDed from a delta and a darkness condition, the red-flash test as a change
in "is this pixel saturated red", the two ORed together.

That correspondence is the point. Every failure this project has measured
traces to the loss and the pass/fail rule measuring different things: 35.66
dB on synthetic data against risk segments still standing on real footage,
amplitude down 60% with the flicker rate down 12%, the model reaching a
median 14.6% of its own conditioning target. A relaxation that drifts from
the Detector's thresholds would reproduce exactly that.

Torch lives here rather than in `detector/` so the classical detection path
keeps its NumPy-only dependency footprint.

The frequency half of `_is_risky` is deliberately not relaxed.
`detector.scoring.WindowedFlashCounter` sets `had_flag = area_ratio > 0.0`,
so a single pixel over threshold flags the whole frame and the frequency
count saturates at the frame rate on real footage -- measured across
wonbon's full length and Cera Khin's first nine seconds. ANDing a term that
is always true would multiply the penalty by one.
"""
import torch
import torch.nn.functional as F

from detector.profiles import ThresholdProfile
from training.mitigator_losses import relative_luminance_torch

_EPS = 1e-6


def _soft_saturated_red(frames_rgb: torch.Tensor, threshold: float, tau: float) -> torch.Tensor:
    """Relaxed `detector.flash._is_saturated_red`: R / (R + G + B) >= threshold."""
    total = frames_rgb.sum(dim=1, keepdim=True).clamp(min=_EPS)
    ratio = frames_rgb[:, 0:1] / total
    return torch.sigmoid((ratio - threshold) / tau)


def soft_flagged_area(
    prev_rgb: torch.Tensor,
    curr_rgb: torch.Tensor,
    profile: ThresholdProfile,
    tau: float = 0.02,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Relaxed flagged-area ratio between two (B, 3, H, W) sRGB frames.

    Returns one scalar per batch item. `valid_mask` is an optional
    (B, 1, H, W) tensor of weights in [0, 1]; pixels weighted 0 are left out
    of the average, which is how the caller discards the replicated border a
    warp leaves behind.
    """
    prev_luminance = relative_luminance_torch(prev_rgb)
    curr_luminance = relative_luminance_torch(curr_rgb)

    delta = (curr_luminance - prev_luminance).abs()
    darker = torch.minimum(prev_luminance, curr_luminance)
    general = torch.sigmoid((delta - profile.general_flash_delta_threshold) / tau) * torch.sigmoid(
        (profile.general_flash_dark_threshold - darker) / tau
    )

    red = (
        _soft_saturated_red(prev_rgb, profile.red_saturation_ratio_threshold, tau)
        - _soft_saturated_red(curr_rgb, profile.red_saturation_ratio_threshold, tau)
    ).abs()

    # `maximum` is the relaxation of the pipeline's `|`: a pixel counts once
    # whether one test fires or both.
    flagged = torch.maximum(general, red)

    if valid_mask is None:
        return flagged.mean(dim=(1, 2, 3))
    weight = valid_mask.sum(dim=(1, 2, 3)).clamp(min=_EPS)
    return (flagged * valid_mask).sum(dim=(1, 2, 3)) / weight


def risk_penalty(
    area: torch.Tensor, max_area_ratio: float, tau_area: float = 0.05
) -> torch.Tensor:
    """Charge only for the part of `area` that exceeds the profile's limit.

    softplus rather than relu so a model sitting just under the limit still
    feels a small pull toward headroom instead of a flat zero it can drift
    back across.
    """
    return F.softplus((area - max_area_ratio) / tau_area) * tau_area
