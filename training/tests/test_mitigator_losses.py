import numpy as np
import pytest
import torch

from training.mitigator_losses import l1_ssim_loss, psnr, ssim, temporal_consistency_loss


def test_ssim_of_identical_images_is_one():
    img = torch.rand(1, 3, 32, 32)
    assert torch.isclose(ssim(img, img), torch.tensor(1.0), atol=1e-4)


def test_ssim_of_very_different_images_is_lower_than_identical():
    img1 = torch.zeros(1, 3, 32, 32)
    img2 = torch.ones(1, 3, 32, 32)
    assert ssim(img1, img2) < ssim(img1, img1)


def test_psnr_of_identical_images_is_very_high():
    img = torch.rand(1, 3, 16, 16)
    assert psnr(img, img) > 40.0


def test_psnr_decreases_as_images_diverge():
    img1 = torch.full((1, 3, 16, 16), 0.5)
    img2_close = img1 + 0.01
    img2_far = img1 + 0.4
    assert psnr(img1, img2_close) > psnr(img1, img2_far)


def test_l1_ssim_loss_is_near_zero_for_identical_images():
    img = torch.rand(1, 3, 16, 16)
    loss = l1_ssim_loss(img, img)
    assert loss.item() < 1e-4


def test_l1_ssim_loss_is_positive_for_different_images():
    img1 = torch.zeros(1, 3, 16, 16)
    img2 = torch.ones(1, 3, 16, 16)
    loss = l1_ssim_loss(img1, img2)
    assert loss.item() > 0.0


def test_l1_ssim_loss_gradients_flow():
    pred = torch.rand(1, 3, 16, 16, requires_grad=True)
    target = torch.rand(1, 3, 16, 16)
    loss = l1_ssim_loss(pred, target)
    loss.backward()
    assert pred.grad is not None


def test_torch_luminance_matches_the_detector_formula():
    # The temporal loss only means something if it measures the same
    # quantity the Detector thresholds on -- a torch reimplementation that
    # drifts from detector.luminance would optimise the wrong target.
    from detector.luminance import relative_luminance
    from training.mitigator_losses import relative_luminance_torch

    rgb = np.random.default_rng(0).random((5, 7, 3)).astype(np.float32)
    expected = relative_luminance(rgb)
    actual = relative_luminance_torch(torch.from_numpy(rgb.transpose(2, 0, 1))[None])
    assert torch.allclose(actual[0, 0], torch.from_numpy(expected), atol=1e-6)


def test_temporal_loss_is_zero_when_prediction_matches_target_pair():
    a = torch.rand(2, 3, 8, 8)
    b = torch.rand(2, 3, 8, 8)
    assert temporal_consistency_loss(a, b, a, b).item() == pytest.approx(0.0, abs=1e-6)


def test_temporal_loss_penalises_flicker_the_target_does_not_have():
    # Target is static between the two frames; the prediction jumps. This
    # is exactly the residual flicker the per-frame L1/SSIM loss cannot see.
    static = torch.full((1, 3, 8, 8), 0.4)
    flickering = torch.full((1, 3, 8, 8), 0.8)
    assert temporal_consistency_loss(static, flickering, static, static).item() > 0.1


def test_temporal_loss_preserves_real_motion_in_the_target():
    # The target itself brightens (a camera move, a genuine cut). A
    # prediction that reproduces that change must NOT be penalised --
    # otherwise the loss would flatten legitimate content along with the
    # flicker.
    dark = torch.full((1, 3, 8, 8), 0.3)
    bright = torch.full((1, 3, 8, 8), 0.9)
    assert temporal_consistency_loss(dark, bright, dark, bright).item() == pytest.approx(0.0, abs=1e-6)


def test_temporal_loss_decreases_as_residual_flicker_shrinks():
    static = torch.full((1, 3, 8, 8), 0.4)
    strong = temporal_consistency_loss(static, torch.full((1, 3, 8, 8), 0.9), static, static)
    weak = temporal_consistency_loss(static, torch.full((1, 3, 8, 8), 0.5), static, static)
    assert weak.item() < strong.item()


def test_temporal_loss_gradients_flow_and_stay_finite_out_of_range():
    # `restored = center + delta` is unclamped, so predictions can leave
    # [0, 1]. The sRGB curve's fractional exponent produces NaN on a
    # negative base, which would silently poison every weight.
    pred_a = torch.full((1, 3, 4, 4), -0.2, requires_grad=True)
    pred_b = torch.full((1, 3, 4, 4), 1.4, requires_grad=True)
    target = torch.full((1, 3, 4, 4), 0.5)
    loss = temporal_consistency_loss(pred_a, pred_b, target, target)
    loss.backward()
    assert torch.isfinite(loss).all()
    assert torch.isfinite(pred_b.grad).all()


def test_warp_by_shift_matches_compensate_shift_for_integer_shifts():
    # The loss must move pixels the same way the Detector does, or it scores
    # a different alignment than the one the pass/fail rule was computed on.
    from detector.motion import compensate_shift
    from training.mitigator_losses import warp_by_shift

    frame = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    expected = compensate_shift(frame, 3.0, -2.0)

    warped, _valid = warp_by_shift(
        torch.from_numpy(frame)[None, None], torch.tensor([3.0]), torch.tensor([-2.0])
    )
    # Compare the interior only; the two disagree at the border by
    # construction (cv2 replicates, grid_sample zero-fills and the validity
    # mask marks it), and that difference is what `valid` exists to express.
    assert np.allclose(warped[0, 0, 4:-4, 4:-4].numpy(), expected[4:-4, 4:-4], atol=1e-4)


def test_warp_by_shift_zero_is_the_identity():
    from training.mitigator_losses import warp_by_shift

    frame = torch.rand(2, 3, 16, 16)
    warped, valid = warp_by_shift(frame, torch.zeros(2), torch.zeros(2))
    assert torch.allclose(warped, frame, atol=1e-5)
    assert valid.min() > 0.99


def test_warp_by_shift_marks_the_invented_border_invalid():
    from training.mitigator_losses import warp_by_shift

    frame = torch.rand(1, 1, 32, 32)
    _warped, valid = warp_by_shift(frame, torch.tensor([6.0]), torch.tensor([0.0]))
    assert valid[0, 0, :, 0].max() < 0.5     # left column came from outside
    assert valid[0, 0, :, -1].min() > 0.5    # right column is real


def test_warp_by_shift_is_differentiable():
    from training.mitigator_losses import warp_by_shift

    frame = torch.rand(1, 3, 16, 16, requires_grad=True)
    warped, _valid = warp_by_shift(frame, torch.tensor([2.5]), torch.tensor([-1.5]))
    warped.sum().backward()
    assert torch.isfinite(frame.grad).all()
    assert frame.grad.abs().sum() > 0


def test_warp_by_shift_handles_per_item_shifts_in_a_batch():
    from training.mitigator_losses import warp_by_shift

    frame = torch.rand(1, 1, 16, 16).repeat(2, 1, 1, 1)
    warped, _valid = warp_by_shift(frame, torch.tensor([0.0, 4.0]), torch.zeros(2))
    assert torch.allclose(warped[0], frame[0], atol=1e-5)
    assert not torch.allclose(warped[1], frame[1], atol=1e-2)
