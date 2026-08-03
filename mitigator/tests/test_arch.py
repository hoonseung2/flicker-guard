import torch

from mitigator.arch import MitigatorNet, mitigate_frame


def test_forward_output_shape_matches_input_resolution():
    model = MitigatorNet()
    window = torch.rand(2, 9, 32, 32)
    mask = torch.rand(2, 1, 32, 32)
    histogram = torch.rand(2, 64)
    out = model(window, mask, histogram)
    assert out.shape == (2, 3, 32, 32)


def test_forward_output_shape_with_resolution_not_a_multiple_of_8():
    # Real inference runs on arbitrary clip resolutions (e.g. 854x480, not
    # divisible by 8) -- the encoder/decoder's downsample-then-upsample must
    # not silently change the output size.
    model = MitigatorNet()
    window = torch.rand(1, 9, 37, 45)
    mask = torch.rand(1, 1, 37, 45)
    histogram = torch.rand(1, 64)
    out = model(window, mask, histogram)
    assert out.shape == (1, 3, 37, 45)


def test_gradients_flow_to_every_parameter():
    model = MitigatorNet()
    window = torch.rand(1, 9, 32, 32)
    mask = torch.rand(1, 1, 32, 32)
    histogram = torch.rand(1, 64)
    out = model(window, mask, histogram)
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
    histogram = torch.rand(1, 64)
    with torch.no_grad():
        result = mitigate_frame(window, mask, histogram, model)
    center = window[:, 3:6, :, :]
    assert torch.equal(result, center)


def test_mitigate_frame_returns_center_frame_shape():
    model = MitigatorNet()
    model.eval()
    window = torch.rand(1, 9, 20, 24)
    mask = torch.ones(1, 1, 20, 24)
    histogram = torch.rand(1, 64)
    with torch.no_grad():
        result = mitigate_frame(window, mask, histogram, model)
    assert result.shape == (1, 3, 20, 24)
