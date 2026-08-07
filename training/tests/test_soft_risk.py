import numpy as np
import pytest
import torch

from detector.flash import red_flash_mask, transition_mask
from detector.luminance import relative_luminance
from detector.profiles import ThresholdProfile
from training.soft_risk import risk_penalty, soft_flagged_area

PROFILE = ThresholdProfile(
    name="test", max_flashes_per_second=3, max_area_ratio=0.25,
    general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
    red_saturation_ratio_threshold=0.80,
)


def _to_torch(frame_hwc):
    return torch.from_numpy(frame_hwc.transpose(2, 0, 1)).float()[None]


def _hard_area(prev_hwc, curr_hwc):
    """The Detector's own flagged-area ratio, computed with numpy."""
    mask = transition_mask(
        relative_luminance(prev_hwc), relative_luminance(curr_hwc),
        dark_threshold=PROFILE.general_flash_dark_threshold,
        delta_threshold=PROFILE.general_flash_delta_threshold,
    ) | red_flash_mask(prev_hwc, curr_hwc,
                       saturation_ratio_threshold=PROFILE.red_saturation_ratio_threshold)
    return float(mask.mean())


def test_identical_frames_have_almost_no_flagged_area():
    frame = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32) * 0.5
    area = soft_flagged_area(_to_torch(frame), _to_torch(frame), PROFILE)
    assert area.shape == (1,)
    assert area.item() < 0.01


def test_soft_area_tracks_the_detector_hard_area():
    # The whole point of the relaxation: it must measure the same quantity
    # the Detector thresholds on. If these diverge, the loss optimises
    # something the pass/fail rule does not care about -- the exact failure
    # this design exists to remove.
    rng = np.random.default_rng(1)
    base = rng.random((64, 64, 3)).astype(np.float32) * 0.2
    for fraction in (0.10, 0.25, 0.50, 0.80):
        bright = base.copy()
        rows = int(64 * fraction)
        bright[:rows] = np.clip(bright[:rows] + 0.7, 0.0, 1.0)

        hard = _hard_area(base, bright)
        soft = soft_flagged_area(_to_torch(base), _to_torch(bright), PROFILE).item()
        assert abs(soft - hard) < 0.05, f"fraction {fraction}: soft {soft:.3f} vs hard {hard:.3f}"


def test_red_flash_is_detected_without_a_luminance_change():
    # red_flash_mask fires on a change in "is this pixel saturated red",
    # independent of brightness. A relaxation that only softened the
    # luminance test would score this pair as safe.
    rng = np.random.default_rng(2)
    grey = rng.random((32, 32, 3)).astype(np.float32) * 0.1 + 0.3
    red = grey.copy()
    red[:, :, 0] = 0.9
    red[:, :, 1] = 0.02
    red[:, :, 2] = 0.02

    assert soft_flagged_area(_to_torch(grey), _to_torch(red), PROFILE).item() > 0.5


def test_gradients_flow_to_both_frames():
    rng = np.random.default_rng(3)
    a = _to_torch(rng.random((16, 16, 3)).astype(np.float32) * 0.3).requires_grad_(True)
    b = _to_torch(rng.random((16, 16, 3)).astype(np.float32) * 0.9).requires_grad_(True)

    soft_flagged_area(a, b, PROFILE).sum().backward()
    assert torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0
    assert torch.isfinite(b.grad).all() and b.grad.abs().sum() > 0


def test_valid_mask_excludes_pixels_from_the_average():
    # The temporal term warps one frame, and the vacated edge holds pixels
    # that have no real correspondence. Those must not count toward the
    # area, or a wide warp dilutes it and the model is under-charged for
    # exceeding the limit.
    #
    # The flagged region and the excluded region are deliberately the SAME
    # half: excluding it must drive the area to roughly zero. A test where
    # they are uncorrelated would pass just as well if valid_mask were
    # ignored entirely.
    rng = np.random.default_rng(4)
    base = rng.random((32, 32, 3)).astype(np.float32) * 0.2
    bright = base.copy()
    bright[:, :16] = np.clip(bright[:, :16] + 0.7, 0.0, 1.0)   # flag the left half

    valid = torch.ones(1, 1, 32, 32)
    valid[:, :, :, :16] = 0.0                                   # exclude the left half

    full = soft_flagged_area(_to_torch(base), _to_torch(bright), PROFILE).item()
    excluded = soft_flagged_area(_to_torch(base), _to_torch(bright), PROFILE, valid_mask=valid).item()

    assert full > 0.4          # about half the frame is genuinely flagged
    assert excluded < 0.05     # and none of it survives the mask


def test_penalty_is_negligible_below_the_limit_and_grows_above():
    below = risk_penalty(torch.tensor([0.10]), max_area_ratio=0.25).item()
    at = risk_penalty(torch.tensor([0.25]), max_area_ratio=0.25).item()
    above = risk_penalty(torch.tensor([0.60]), max_area_ratio=0.25).item()

    assert below < 0.01
    assert below < at < above
    assert above > 0.3


def test_penalty_keeps_a_gradient_below_the_limit():
    # softplus rather than relu: a model sitting just under the limit still
    # feels a small pull toward more headroom, instead of a flat zero that
    # lets it drift back over.
    area = torch.tensor([0.10], requires_grad=True)
    risk_penalty(area, max_area_ratio=0.25).backward()
    assert area.grad.abs().item() > 0
