"""MitigatorNet: predicts a residual correction for the center frame of a
3-frame degraded window, conditioned on a per-pixel risk mask and a
required-strength scalar (both from prior.compute.compute_prior).

Residual, not full-frame, prediction: degraded and clean frames are
identical outside the injected/risky region by construction (DatasetSynth
only ever modifies pixels inside the mask), so predicting "what changed"
is an easier target than reconstructing the whole frame from scratch, and
naturally biases the untrained network toward leaving already-clean pixels
alone. mitigate_frame's mask blend then makes that bias a hard guarantee
rather than a hope: pixels outside the mask are the original input,
unconditionally, no matter what the network outputs.

Architecture is a small U-Net with a self-attention bottleneck -- inspired
by Flickerformer's "3-frame window -> lightweight Restormer-style network"
idea (README section 1), reimplemented from scratch rather than adapted
from Flickerformer's published code, which ships with no LICENSE file.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = _ConvBlock(in_channels, out_channels)
        self.pool = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.block(x)
        down = self.pool(skip)
        return down, skip


class _Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.block = _ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            # Odd input resolutions round differently on the way down vs
            # back up (e.g. 37 -> 19 -> 37 is fine, but 37 -> 19 -> 38 is
            # not) -- nudge back to the skip connection's exact size rather
            # than fail on non-power-of-2-friendly resolutions.
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class _BottleneckAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        normed = self.norm(x)
        tokens = normed.flatten(2).transpose(1, 2)  # (B, H*W, C)
        attended, _ = self.attn(tokens, tokens, tokens)
        attended = attended.transpose(1, 2).reshape(b, c, h, w)
        return x + attended


class MitigatorNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 RGB frames + risk mask + required strength.
        #
        # The strength replaces a 64-bin illumination histogram. That
        # histogram described what normal illumination looks like, derived
        # from the same frame's non-masked pixels -- sound on synthetic data,
        # where shapes are pasted onto ordinary video, but wrong on real
        # concert footage, where inside the mask is stage lighting and
        # outside is a dark crowd. Two different objects. It asked the model
        # to make the lights as dark as the audience, and the model followed
        # it to a median 14.6%.
        in_channels = 3 * 3 + 1 + 1
        self.down1 = _Down(in_channels, 32)
        self.down2 = _Down(32, 64)
        self.down3 = _Down(64, 128)
        self.bottleneck = _BottleneckAttention(128)
        self.up3 = _Up(128, 128, 64)
        self.up2 = _Up(64, 64, 32)
        self.up1 = _Up(32, 32, 32)
        self.out_conv = nn.Conv2d(32, 3, 3, padding=1)

    def forward(
        self,
        window: torch.Tensor,
        mask: torch.Tensor,
        strength: torch.Tensor,
    ) -> torch.Tensor:
        b, _, h, w = window.shape
        strength_spatial = strength.view(b, 1, 1, 1).expand(-1, -1, h, w)
        x = torch.cat([window, mask, strength_spatial], dim=1)

        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        x3 = self.bottleneck(x3)
        x = self.up3(x3, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        return self.out_conv(x)


# Kept in step with detector.luminance, which is the authority. Transcribed
# rather than imported because detector/ is deliberately torch-free, and
# imported from neither training/ nor mitigator_losses because inference
# must not depend on the training package.
_SRGB_A = 0.055
_SRGB_ENCODED_CUTOFF = 0.04045
_SRGB_LINEAR_CUTOFF = 0.0031308
_R_COEFF, _G_COEFF, _B_COEFF = 0.2126, 0.7152, 0.0722
# Below this linear luminance a pixel is black enough to have no meaningful
# chromaticity, and target/current would divide by nearly nothing.
_LUM_FLOOR = 1e-4


def _srgb_to_linear_torch(x: torch.Tensor) -> torch.Tensor:
    safe = x.clamp(min=0.0)
    return torch.where(safe <= _SRGB_ENCODED_CUTOFF, safe / 12.92,
                       ((safe + _SRGB_A) / (1 + _SRGB_A)) ** 2.4)


def _linear_to_srgb_torch(x: torch.Tensor) -> torch.Tensor:
    safe = x.clamp(min=0.0)
    return torch.where(safe <= _SRGB_LINEAR_CUTOFF, safe * 12.92,
                       (1 + _SRGB_A) * safe ** (1 / 2.4) - _SRGB_A)


def _luminance_of(linear_rgb: torch.Tensor) -> torch.Tensor:
    return (_R_COEFF * linear_rgb[:, 0:1]
            + _G_COEFF * linear_rgb[:, 1:2]
            + _B_COEFF * linear_rgb[:, 2:3])


def apply_luminance_of(center: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Give `center` the luminance of `target` without moving its colour.

    The model predicts an additive residual, and every loss term scores it
    through `relative_luminance`, whose weights are 0.2126 / 0.7152 / 0.0722.
    The minimum-norm way to hit a luminance target is therefore to move each
    channel in proportion to its weight -- which is what the model learned,
    and which visibly shifts hue, because subtracting a fixed-direction
    vector is not the same as scaling a pixel toward black. Measured on
    wonbon: per-pixel |delta| ran G:R:B = 10.1:2.8:1 against luminance
    coefficients of 9.9:2.9:1, and the output went magenta.

    Scaling linear RGB by a scalar leaves chromaticity exactly unchanged, so
    taking only the luminance the model asked for and applying it this way
    makes the hue shift impossible rather than merely penalised -- and needs
    no retraining, since the terms that decide the Detector verdict read
    luminance only.

    The scaling has to happen in linear light. Multiplying encoded sRGB
    values would shift colour again through the gamma curve.
    """
    linear_c = _srgb_to_linear_torch(center)
    lum_c = _luminance_of(linear_c)
    lum_t = _luminance_of(_srgb_to_linear_torch(target))

    scaled = _linear_to_srgb_torch(linear_c * (lum_t / lum_c.clamp(min=_LUM_FLOOR)))
    # A near-black pixel has no colour to preserve and an unstable ratio;
    # fall back to the additive result there rather than amplifying noise.
    return torch.where(lum_c > _LUM_FLOOR, scaled, target)


def mitigate_frame(
    window: torch.Tensor,
    mask: torch.Tensor,
    strength: torch.Tensor,
    model: MitigatorNet,
    preserve_hue: bool = False,
) -> torch.Tensor:
    delta = model(window, mask, strength)
    center = window[:, 3:6, :, :]
    restored = center + delta
    if preserve_hue:
        restored = apply_luminance_of(center, restored)
    return mask * restored + (1 - mask) * center
