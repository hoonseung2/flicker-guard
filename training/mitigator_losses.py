"""Differentiable SSIM/PSNR and the combined L1+SSIM training loss for
Mitigator. Hand-implemented rather than pulled from torchmetrics: SSIM must
be differentiable to use as a training loss (not just an eval-time metric),
and the standard windowed formulation (Wang et al. 2004) is short enough
that adding a new dependency isn't worth it.
"""
import torch
import torch.nn.functional as F

_C1 = 0.01 ** 2
_C2 = 0.03 ** 2


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
