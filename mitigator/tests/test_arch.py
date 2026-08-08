import torch

from mitigator.arch import MitigatorNet, mitigate_frame


def test_forward_output_shape_matches_input_resolution():
    model = MitigatorNet()
    window = torch.rand(2, 9, 32, 32)
    mask = torch.rand(2, 1, 32, 32)
    strength = torch.rand(2, 1)
    out = model(window, mask, strength)
    assert out.shape == (2, 3, 32, 32)


def test_forward_output_shape_with_resolution_not_a_multiple_of_8():
    # Real inference runs on arbitrary clip resolutions (e.g. 854x480, not
    # divisible by 8) -- the encoder/decoder's downsample-then-upsample must
    # not silently change the output size.
    model = MitigatorNet()
    window = torch.rand(1, 9, 37, 45)
    mask = torch.rand(1, 1, 37, 45)
    strength = torch.rand(1, 1)
    out = model(window, mask, strength)
    assert out.shape == (1, 3, 37, 45)


def test_gradients_flow_to_every_parameter():
    model = MitigatorNet()
    window = torch.rand(1, 9, 32, 32)
    mask = torch.rand(1, 1, 32, 32)
    strength = torch.rand(1, 1)
    out = model(window, mask, strength)
    out.sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient reached parameter {name}"


def test_mitigate_frame_preserves_pixels_outside_mask_exactly():
    # The core safety guarantee: wherever mask == 0, the output must be
    # bit-for-bit the input, regardless of what an untrained (effectively
    # random) network predicts.
    model = MitigatorNet()
    model.eval()
    window = torch.rand(1, 9, 16, 16)
    mask = torch.zeros(1, 1, 16, 16)
    strength = torch.rand(1, 1)
    with torch.no_grad():
        result = mitigate_frame(window, mask, strength, model)
    center = window[:, 3:6, :, :]
    assert torch.equal(result, center)


def test_mitigate_frame_returns_center_frame_shape():
    model = MitigatorNet()
    model.eval()
    window = torch.rand(1, 9, 20, 24)
    mask = torch.ones(1, 1, 20, 24)
    strength = torch.rand(1, 1)
    with torch.no_grad():
        result = mitigate_frame(window, mask, strength, model)
    assert result.shape == (1, 3, 20, 24)


def test_forward_accepts_a_strength_scalar_instead_of_a_histogram():
    from mitigator.arch import MitigatorNet

    model = MitigatorNet()
    window = torch.rand(2, 9, 32, 32)
    mask = torch.ones(2, 1, 32, 32)
    strength = torch.tensor([[0.3], [0.9]])
    assert model(window, mask, strength).shape == (2, 3, 32, 32)


def test_strength_changes_the_output():
    # Conditioning the model on a value it ignores would be worse than not
    # conditioning it at all -- it would look informed and behave otherwise.
    from mitigator.arch import MitigatorNet

    torch.manual_seed(0)
    model = MitigatorNet()
    window = torch.rand(1, 9, 32, 32)
    mask = torch.ones(1, 1, 32, 32)

    low = model(window, mask, torch.tensor([[0.1]]))
    high = model(window, mask, torch.tensor([[0.9]]))
    assert not torch.allclose(low, high, atol=1e-5)


def test_mitigate_frame_takes_strength():
    from mitigator.arch import MitigatorNet, mitigate_frame

    model = MitigatorNet()
    window = torch.rand(1, 9, 16, 16)
    mask = torch.zeros(1, 1, 16, 16)
    result = mitigate_frame(window, mask, torch.tensor([[0.5]]), model)
    # mask is all zeros, so the hard blend must return the centre frame
    # untouched regardless of what the network produced.
    assert torch.allclose(result, window[:, 3:6], atol=1e-6)


def test_strength_is_isolated_per_batch_item():
    # Each item's strength is broadcast spatially before concatenation, and
    # the network has no batch-mixing layer (GroupNorm(1, ...) normalises
    # per-sample, MultiheadAttention attends only within each sample's own
    # H*W tokens) -- so item 0's output must be identical whether it's run
    # alone or alongside item 1 conditioned on a different strength. This
    # pins that down explicitly rather than leaving it to be checked by hand.
    from mitigator.arch import MitigatorNet

    torch.manual_seed(0)
    model = MitigatorNet()
    model.eval()
    window = torch.rand(2, 9, 32, 32)
    mask = torch.ones(2, 1, 32, 32)
    strength = torch.tensor([[0.2], [0.8]])

    with torch.no_grad():
        batched = model(window, mask, strength)
        solo_0 = model(window[:1], mask[:1], strength[:1])
        solo_1 = model(window[1:], mask[1:], strength[1:])

    assert torch.allclose(batched[:1], solo_0, atol=1e-5)
    assert torch.allclose(batched[1:], solo_1, atol=1e-5)


def test_apply_luminance_of_preserves_chromaticity_exactly():
    # The whole point: the output pixel is the input pixel scaled toward
    # black, so its colour is untouched no matter how far luminance moves.
    #
    # Checked in LINEAR light, which is where chromaticity is defined.
    # Encoded sRGB ratios are deliberately not preserved -- a scale followed
    # by re-encoding cannot preserve both, and it is the linear ratio that
    # determines the colour a viewer sees.
    from mitigator.arch import apply_luminance_of, _srgb_to_linear_torch

    torch.manual_seed(0)
    center = torch.rand(2, 3, 8, 8) * 0.8 + 0.1
    target = (center * 0.4).clamp(0, 1)  # much darker, same colour

    out = apply_luminance_of(center, target)

    lin_in = _srgb_to_linear_torch(center)
    lin_out = _srgb_to_linear_torch(out)
    ratio_in = lin_in / lin_in.sum(dim=1, keepdim=True)
    ratio_out = lin_out / lin_out.sum(dim=1, keepdim=True)
    assert torch.allclose(ratio_in, ratio_out, atol=1e-4)


def test_apply_luminance_of_matches_the_requested_luminance():
    from mitigator.arch import apply_luminance_of, _luminance_of, _srgb_to_linear_torch

    torch.manual_seed(1)
    center = torch.rand(1, 3, 8, 8) * 0.8 + 0.15
    target = (center * 0.5).clamp(0, 1)

    out = apply_luminance_of(center, target)

    lum = lambda x: _luminance_of(_srgb_to_linear_torch(x))
    assert torch.allclose(lum(out), lum(target), atol=1e-4)


def test_apply_luminance_of_falls_back_to_the_additive_result_on_black():
    # A black pixel has no chromaticity to preserve and the luminance ratio
    # would divide by nearly zero, amplifying noise into colour.
    from mitigator.arch import apply_luminance_of

    center = torch.zeros(1, 3, 4, 4)
    target = torch.full((1, 3, 4, 4), 0.3)

    assert torch.equal(apply_luminance_of(center, target), target)


def test_mitigate_frame_preserves_hue_by_default():
    # Default since 2026-08-09. The additive form stays reachable only to
    # reproduce measurements taken before that.
    from mitigator.arch import MitigatorNet, mitigate_frame

    torch.manual_seed(2)
    model = MitigatorNet()
    window = torch.rand(1, 9, 16, 16)
    mask = torch.ones(1, 1, 16, 16)
    strength = torch.rand(1, 1)

    with torch.no_grad():
        default = mitigate_frame(window, mask, strength, model)
        hue = mitigate_frame(window, mask, strength, model, preserve_hue=True)
        additive = mitigate_frame(window, mask, strength, model, preserve_hue=False)

    assert torch.equal(default, hue)
    assert not torch.allclose(default, additive)
