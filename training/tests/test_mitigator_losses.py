import torch

from training.mitigator_losses import l1_ssim_loss, psnr, ssim


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
