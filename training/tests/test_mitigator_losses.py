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


def _profile():
    from detector.profiles import ThresholdProfile
    return ThresholdProfile(
        name="test", max_flashes_per_second=3, max_area_ratio=0.25,
        general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
        red_saturation_ratio_threshold=0.80,
    )


def _flickering_pair(seed=0):
    """Input frames: a dark frame and a much brighter one. Pure flicker."""
    g = torch.Generator().manual_seed(seed)
    dark = torch.rand(1, 3, 32, 32, generator=g) * 0.2
    bright = (dark + 0.7).clamp(0, 1)
    return dark, bright


def test_passing_the_input_through_unchanged_costs_zero_fidelity():
    from training.mitigator_losses import self_supervised_loss

    in_a, in_b = _flickering_pair()
    terms = self_supervised_loss(in_a, in_b, in_a, in_b, torch.zeros(1), torch.zeros(1), _profile())
    assert terms["fidelity"].item() == pytest.approx(0.0, abs=1e-6)


def test_the_do_nothing_output_is_still_charged_for_risk():
    # This is the whole design: passing the input straight through is free
    # on fidelity and free on nothing else. If the risk term did not fire
    # here, the model's optimum would be to do nothing.
    #
    # It is not enough to check that terms["risk"] is nonzero -- that only
    # proves the value is computed and returned under its key, not that it
    # is actually charged into the total. Isolate risk's contribution to
    # "loss" directly: this fails if lambda_risk * risk is ever dropped
    # from the sum, even though risk itself would still read > 0.05.
    from training.mitigator_losses import self_supervised_loss

    in_a, in_b = _flickering_pair()
    lt, lr = 1.0, 1.0
    terms = self_supervised_loss(in_a, in_b, in_a, in_b, torch.zeros(1), torch.zeros(1),
                                 _profile(), lambda_temporal=lt, lambda_risk=lr)
    assert terms["risk"].item() > 0.05
    assert terms["temporal"].item() > 0.05
    risk_contribution = terms["loss"] - terms["fidelity"] - lt * terms["temporal"]
    assert risk_contribution.item() == pytest.approx((lr * terms["risk"]).item(), abs=1e-6)
    assert terms["loss"].item() > terms["fidelity"].item()


def test_flattening_the_pair_clears_the_risk_term_but_costs_fidelity():
    from training.mitigator_losses import self_supervised_loss

    in_a, in_b = _flickering_pair()
    flat = (in_a + in_b) / 2
    terms = self_supervised_loss(flat, flat, in_a, in_b, torch.zeros(1), torch.zeros(1), _profile())

    assert terms["risk"].item() < 0.01
    assert terms["temporal"].item() < 0.01
    assert terms["fidelity"].item() > 0.05


def test_lambda_risk_zero_removes_the_risk_term_entirely():
    # The pre-design objective, for regression comparison.
    from training.mitigator_losses import self_supervised_loss

    in_a, in_b = _flickering_pair()
    terms = self_supervised_loss(in_a, in_b, in_a, in_b, torch.zeros(1), torch.zeros(1),
                                 _profile(), lambda_risk=0.0)
    assert terms["loss"].item() == pytest.approx(
        (terms["fidelity"] + terms["temporal"]).item(), abs=1e-6
    )


def test_the_shift_is_applied_before_the_temporal_comparison():
    # A pure pan is not flicker. With the correct shift the two frames align
    # and the temporal term collapses; without it the same pair looks like a
    # large change. If this fails, the loss will fight camera movement.
    from training.mitigator_losses import self_supervised_loss

    g = torch.Generator().manual_seed(7)
    a = torch.rand(1, 3, 48, 48, generator=g) * 0.5 + 0.2
    b = torch.roll(a, shifts=4, dims=3)

    compensated = self_supervised_loss(a, b, a, b, torch.tensor([4.0]), torch.zeros(1), _profile())
    ignored = self_supervised_loss(a, b, a, b, torch.zeros(1), torch.zeros(1), _profile())
    assert compensated["temporal"].item() < ignored["temporal"].item() * 0.5


def test_all_terms_produce_finite_gradients():
    from training.mitigator_losses import self_supervised_loss

    in_a, in_b = _flickering_pair(seed=3)
    out_a = (in_a * 0.8).clone().requires_grad_(True)
    out_b = (in_b * 0.8).clone().requires_grad_(True)

    self_supervised_loss(out_a, out_b, in_a, in_b, torch.zeros(1), torch.zeros(1),
                         _profile())["loss"].backward()
    for grad in (out_a.grad, out_b.grad):
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


def test_reported_terms_sum_to_the_reported_loss():
    # The training loop logs these separately to watch the fidelity/risk
    # trade-off; a reported breakdown that does not reconstruct the total
    # would make that log misleading.
    from training.mitigator_losses import self_supervised_loss

    in_a, in_b = _flickering_pair(seed=5)
    lt, lr = 0.7, 1.3
    terms = self_supervised_loss(in_a * 0.9, in_b * 0.9, in_a, in_b,
                                 torch.zeros(1), torch.zeros(1), _profile(),
                                 lambda_temporal=lt, lambda_risk=lr)
    expected = terms["fidelity"] + lt * terms["temporal"] + lr * terms["risk"]
    assert terms["loss"].item() == pytest.approx(expected.item(), abs=1e-6)


def test_per_item_shift_is_forwarded_correctly_in_a_batch():
    # Guards the forwarding path from self_supervised_loss into warp_by_shift:
    # item 0 needs dx=0, item 1 needs dx=4. If a single shared shift were
    # used instead of the per-item tensor, at least one item would be
    # misaligned and its terms would not collapse.
    from training.mitigator_losses import self_supervised_loss

    g = torch.Generator().manual_seed(11)
    a0 = torch.rand(1, 3, 48, 48, generator=g) * 0.5 + 0.2
    b0 = a0.clone()
    a1 = torch.rand(1, 3, 48, 48, generator=g) * 0.5 + 0.2
    b1 = torch.roll(a1, shifts=4, dims=3)

    a = torch.cat([a0, a1], dim=0)
    b = torch.cat([b0, b1], dim=0)
    dx = torch.tensor([0.0, 4.0])
    dy = torch.zeros(2)

    correct = self_supervised_loss(a, b, a, b, dx, dy, _profile())
    assert correct["temporal"].item() < 0.01
    assert correct["risk"].item() < 0.01

    # A single shared shift (e.g. the item-0 value applied to both) leaves
    # item 1 unaligned, so the batch-averaged terms come out materially
    # higher.
    shared_wrong = self_supervised_loss(a, b, a, b, torch.zeros(2), dy, _profile())
    assert shared_wrong["temporal"].item() > correct["temporal"].item() + 0.05


def test_fidelity_and_temporal_average_per_item_not_pooled_across_the_batch():
    # fidelity and temporal must use the SAME reduction convention as risk
    # and soft_area: average per batch item, then average across items.
    #
    # Item 0 has no shift (fully valid) and a large residual flicker the
    # model introduced (out_b0 jumps far from in_b0, with nothing in the
    # input pair to justify it). Item 1 has a real, large (20px) shift that
    # warp_by_shift compensates for exactly, so its temporal delta is ~0,
    # but the shift also zero-fills and invalidates a wide border strip,
    # leaving it with well under item 0's valid-pixel count (1344 vs. 2304
    # of 2304 pixels, measured below) -- large enough that pooled vs.
    # per-item averaging give visibly different numbers, not a difference
    # lost in floating-point noise.
    #
    # Pooling raw elements across the batch would blend item 0's large
    # per-pixel delta over BOTH items' valid pixels (item 0's plus item 1's
    # smaller valid count), diluting it. Averaging per item first weights
    # item 0's average equally with item 1's zero, regardless of how many
    # valid pixels either contributed. The two conventions give visibly
    # different numbers here, which is what this test pins down.
    import torch.nn.functional as F

    from training.mitigator_losses import relative_luminance_torch, self_supervised_loss, warp_by_shift

    g = torch.Generator().manual_seed(21)
    base0 = torch.rand(1, 3, 48, 48, generator=g) * 0.3 + 0.2
    a0, b0 = base0.clone(), base0.clone()
    out_a0 = a0.clone()
    out_b0 = (b0 + 0.4).clamp(0, 1)  # residual flicker with no basis in the input pair

    base1 = torch.rand(1, 3, 48, 48, generator=g) * 0.3 + 0.2
    a1 = base1.clone()
    b1 = torch.roll(base1, shifts=20, dims=3)  # real motion, correctly compensated below
    out_a1, out_b1 = a1.clone(), b1.clone()

    in_a = torch.cat([a0, a1], dim=0)
    in_b = torch.cat([b0, b1], dim=0)
    out_a = torch.cat([out_a0, out_a1], dim=0)
    out_b = torch.cat([out_b0, out_b1], dim=0)
    dx = torch.tensor([0.0, 20.0])
    dy = torch.zeros(2)

    terms = self_supervised_loss(out_a, out_b, in_a, in_b, dx, dy, _profile())

    fid_per_item = (
        F.l1_loss(out_a, in_a, reduction="none").mean(dim=(1, 2, 3))
        + F.l1_loss(out_b, in_b, reduction="none").mean(dim=(1, 2, 3))
    )
    expected_fidelity = fid_per_item.mean()
    assert terms["fidelity"].item() == pytest.approx(expected_fidelity.item(), abs=1e-6)

    warped, valid = warp_by_shift(out_a, dx, dy)
    delta = (relative_luminance_torch(warped) - relative_luminance_torch(out_b)).abs()
    per_item_denom = valid.sum(dim=(1, 2, 3)).clamp(min=1e-6)
    expected_temporal = ((delta * valid).sum(dim=(1, 2, 3)) / per_item_denom).mean()
    assert terms["temporal"].item() == pytest.approx(expected_temporal.item(), abs=1e-6)

    # Confirm the per-item value actually differs -- materially, not just
    # past floating-point noise -- from the pooled-across-the-batch value,
    # so this test would fail (not vacuously pass) if a future change
    # switched the convention back. (Measured: per-item ~0.213, pooled
    # ~0.269 -- a >20% relative gap driven by item 1's smaller valid-pixel
    # count, 1344 vs. item 0's 2304.)
    pooled_denom = valid.sum().clamp(min=1e-6)
    pooled_temporal = ((delta * valid).sum() / pooled_denom).item()
    assert abs(terms["temporal"].item() - pooled_temporal) > 0.03


def test_flattening_with_a_tight_area_limit_still_charges_risk():
    # With max_area_ratio tightened to near zero, even the flattened pair
    # (which clears a normal limit almost entirely -- see
    # test_flattening_the_pair_clears_the_risk_term_but_costs_fidelity)
    # exceeds it, so softplus's small-excess branch should still produce a
    # positive penalty. Regression guard for that branch.
    from detector.profiles import ThresholdProfile
    from training.mitigator_losses import self_supervised_loss

    in_a, in_b = _flickering_pair()
    flat = (in_a + in_b) / 2
    tight_profile = ThresholdProfile(
        name="tight", max_flashes_per_second=3, max_area_ratio=0.001,
        general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
        red_saturation_ratio_threshold=0.80,
    )
    terms = self_supervised_loss(flat, flat, in_a, in_b, torch.zeros(1), torch.zeros(1),
                                 tight_profile)
    assert terms["risk"].item() > 0.0
