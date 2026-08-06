import pytest
import torch

from mitigator.arch import MitigatorNet
from training.mitigator_losses import l1_ssim_loss
from training.train_mitigator import load_checkpoint, main, save_checkpoint, train_one_epoch, _make_patch_collate_fn


class _FixedBatchDataset(torch.utils.data.Dataset):
    """A dataset that always returns the same single example -- lets a
    training-loop test check "does loss go down" without needing real
    data/synthetic/ content or a GPU."""

    def __init__(self, h=16, w=16, n=4):
        torch.manual_seed(0)
        self.window = torch.rand(9, h, w)
        self.mask = torch.ones(1, h, w)
        self.histogram = torch.rand(64)
        self.clean = torch.rand(3, h, w)
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "window": self.window,
            "mask": self.mask,
            "histogram": self.histogram,
            "clean": self.clean,
            # compute_losses restores the centre frame and its temporal
            # partner together, so every example must carry both.
            "window_partner": self.window,
            "mask_partner": self.mask,
            "histogram_partner": self.histogram,
            "clean_partner": self.clean,
        }


def test_train_one_epoch_reduces_loss_over_many_epochs():
    torch.manual_seed(0)
    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    dataloader = torch.utils.data.DataLoader(_FixedBatchDataset(), batch_size=4)

    first_epoch_loss = train_one_epoch(model, optimizer, dataloader, device="cpu")
    last_epoch_loss = first_epoch_loss
    for _ in range(19):
        last_epoch_loss = train_one_epoch(model, optimizer, dataloader, device="cpu")

    assert last_epoch_loss < first_epoch_loss


def test_train_one_epoch_returns_finite_loss():
    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataloader = torch.utils.data.DataLoader(_FixedBatchDataset(), batch_size=4)
    loss = train_one_epoch(model, optimizer, dataloader, device="cpu")
    assert loss == loss  # NaN check (NaN != NaN)
    assert loss != float("inf")


def test_checkpoint_save_and_load_restores_model_and_optimizer_state(tmp_path):
    torch.manual_seed(0)
    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataloader = torch.utils.data.DataLoader(_FixedBatchDataset(), batch_size=4)
    train_one_epoch(model, optimizer, dataloader, device="cpu")
    train_one_epoch(model, optimizer, dataloader, device="cpu")

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, optimizer, epoch=2, best_val_loss=0.5)

    new_model = MitigatorNet()
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    restored_epoch, restored_best_val_loss = load_checkpoint(checkpoint_path, new_model, new_optimizer)

    assert restored_epoch == 2
    assert restored_best_val_loss == 0.5
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)

    orig_state = list(optimizer.state.values())[0]
    new_state = list(new_optimizer.state.values())[0]
    assert torch.equal(orig_state["exp_avg"], new_state["exp_avg"])


def test_load_checkpoint_defaults_best_val_loss_to_inf_for_legacy_checkpoints(tmp_path):
    # A checkpoint written before best_val_loss was tracked has no such key
    # -- load_checkpoint must not raise, and must fall back to inf so a
    # resumed run's first epoch can still win against it, same as a fresh run.
    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "legacy_checkpoint.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": 0}, checkpoint_path)

    new_model = MitigatorNet()
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    _, best_val_loss = load_checkpoint(checkpoint_path, new_model, new_optimizer)

    assert best_val_loss == float("inf")


def test_checkpoint_resume_round_trip_does_not_raise_on_cpu(tmp_path):
    # CPU can't reproduce the device-mismatch RuntimeError this guards
    # against (see the CUDA variant below), but it does catch a regression
    # to resume raising for any other reason -- e.g. a checkpoint/optimizer
    # API mismatch.
    torch.manual_seed(0)
    model = MitigatorNet()
    model.to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataloader = torch.utils.data.DataLoader(_FixedBatchDataset(), batch_size=4)
    train_one_epoch(model, optimizer, dataloader, device="cpu")

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, optimizer, epoch=0, best_val_loss=0.5)

    new_model = MitigatorNet()
    new_model.to("cpu")
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    load_checkpoint(checkpoint_path, new_model, new_optimizer)

    for param in new_model.parameters():
        assert param.device.type == "cpu"
    for state in new_optimizer.state.values():
        assert state["exp_avg"].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_checkpoint_resume_round_trip_keeps_optimizer_state_on_model_device_gpu(tmp_path):
    # Regression test for the --resume device-mismatch bug: main() used to
    # build the optimizer over CPU params and move the model to the GPU only
    # inside train_one_epoch, so a resumed AdamW's exp_avg/exp_avg_sq stayed
    # on CPU while gradients were computed on CUDA -- load_checkpoint would
    # then raise a device-mismatch RuntimeError on the next optimizer.step().
    torch.manual_seed(0)
    model = MitigatorNet()
    model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataloader = torch.utils.data.DataLoader(_FixedBatchDataset(), batch_size=4)
    train_one_epoch(model, optimizer, dataloader, device="cuda")

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, optimizer, epoch=0, best_val_loss=0.5)

    new_model = MitigatorNet()
    new_model.to("cuda")
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    load_checkpoint(checkpoint_path, new_model, new_optimizer)

    for param in new_model.parameters():
        assert param.device.type == "cuda"
    for state in new_optimizer.state.values():
        assert state["exp_avg"].device.type == "cuda"

    # The real bug surfaced on the next optimizer.step() after resume, not
    # on load itself -- exercise one more training step to be sure.
    train_one_epoch(new_model, new_optimizer, dataloader, device="cuda")


class _FakeMitigatorDataset(torch.utils.data.Dataset):
    """Stands in for training.mitigator_dataset.MitigatorDataset in main()
    tests -- avoids needing a real data-dir with synthesized frames/priors
    on disk just to exercise main()'s checkpoint bookkeeping."""

    def __init__(self, *args, **kwargs):
        self.window = torch.rand(9, 8, 8)
        self.mask = torch.ones(1, 8, 8)
        self.histogram = torch.rand(64)
        self.clean = torch.rand(3, 8, 8)
        self.samples_scanned = 1
        self.samples_contributing = 1

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {
            "window": self.window,
            "mask": self.mask,
            "histogram": self.histogram,
            "clean": self.clean,
            # compute_losses restores the centre frame and its temporal
            # partner together, so every example must carry both.
            "window_partner": self.window,
            "mask_partner": self.mask,
            "histogram_partner": self.histogram,
            "clean_partner": self.clean,
        }


def test_main_resume_does_not_clobber_best_pt_with_worse_checkpoint(tmp_path, monkeypatch):
    # Regression test for the residual best.pt-clobber bug: latest.pt used
    # to be saved with the PRE-update best_val_loss, so a --resume run
    # restored a stale, one-epoch-behind threshold. A later epoch that beat
    # that stale threshold but not the true best would then pass the `<`
    # check and overwrite best.pt with an inferior checkpoint.
    import training.train_mitigator as train_mitigator_module

    monkeypatch.setattr(train_mitigator_module, "MitigatorDataset", _FakeMitigatorDataset)

    # epoch0=0.5, epoch1=0.2 (true best), epoch2=0.3 -- worse than the true
    # best (0.2) but better than the stale threshold (0.5) a buggy resume
    # would have restored.
    scripted_losses = iter([0.5, 0.2, 0.3])

    def _fake_evaluate(model, dataloader, device, lambda_temporal=0.0):
        return {"loss": next(scripted_losses), "reconstruction": 0.0, "temporal": 0.0, "psnr": 0.0}

    monkeypatch.setattr(train_mitigator_module, "evaluate", _fake_evaluate)

    checkpoint_dir = tmp_path / "ckpt"
    common_args = [
        "--data-dir", str(tmp_path / "data"),
        "--profile", "configs/profiles/itu.json",
        "--checkpoint-dir", str(checkpoint_dir),
        "--batch-size", "4",
    ]

    main(common_args + ["--epochs", "2"])
    best_after_run1 = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=True)
    assert best_after_run1["epoch"] == 1
    assert best_after_run1["best_val_loss"] == 0.2

    main(common_args + ["--epochs", "3", "--resume"])
    best_after_resume = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=True)
    assert best_after_resume["epoch"] == 1
    assert best_after_resume["best_val_loss"] == 0.2


class _VariableShapeMitigatorDataset(torch.utils.data.Dataset):
    """Val split stand-in with heterogeneous per-item spatial shapes, like
    real DAVIS clips of differing native resolution. Regression fixture for
    the val batch_size=1 fix -- batch_size>1 with collate_fn=None would try
    to torch.stack these mismatched shapes and raise a RuntimeError."""

    _SHAPES = [(8, 8), (8, 12)]

    def __init__(self, *args, **kwargs):
        self.samples_scanned = 1
        self.samples_contributing = 1

    def __len__(self):
        return len(self._SHAPES)

    def __getitem__(self, idx):
        h, w = self._SHAPES[idx]
        return {
            "window": torch.rand(9, h, w),
            "mask": torch.ones(1, h, w),
            "histogram": torch.rand(64),
            "clean": torch.rand(3, h, w),
            "window_partner": torch.rand(9, h, w),
            "mask_partner": torch.ones(1, h, w),
            "histogram_partner": torch.rand(64),
            "clean_partner": torch.rand(3, h, w),
        }


def test_main_val_loader_handles_differently_shaped_examples(tmp_path, monkeypatch):
    # Regression test: real source clips (e.g. DAVIS) don't all share one
    # resolution, and val is deliberately never cropped -- a val batch size
    # > 1 with collate_fn=None used to try to torch.stack mismatched shapes
    # and crash with "stack expects each tensor to be equal size". main()
    # must hardcode val batch_size=1 to sidestep this.
    import training.train_mitigator as train_mitigator_module

    def _fake_dataset(data_dir, profile, split):
        if split == "train":
            return _FakeMitigatorDataset()
        return _VariableShapeMitigatorDataset()

    monkeypatch.setattr(train_mitigator_module, "MitigatorDataset", _fake_dataset)

    checkpoint_dir = tmp_path / "ckpt"
    exit_code = main([
        "--data-dir", str(tmp_path / "data"),
        "--profile", "configs/profiles/itu.json",
        "--checkpoint-dir", str(checkpoint_dir),
        "--batch-size", "4",
        "--epochs", "1",
    ])

    assert exit_code == 0
    assert (checkpoint_dir / "best.pt").exists()


def test_train_one_epoch_raises_clearly_on_nan_loss(monkeypatch):
    # A NaN loss must never be swallowed -- it means something upstream
    # (bad data, exploding lr, etc.) is broken, and silently continuing
    # would corrupt every weight via backward() on NaN.
    import training.train_mitigator as train_mitigator_module

    def _nan_loss(pred, target):
        return pred.sum() * float("nan")

    monkeypatch.setattr(train_mitigator_module, "l1_ssim_loss", _nan_loss)

    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataloader = torch.utils.data.DataLoader(_FixedBatchDataset(), batch_size=4)

    with pytest.raises(RuntimeError, match="NaN"):
        train_one_epoch(model, optimizer, dataloader, device="cpu")


def test_patch_collate_fn_crops_window_mask_clean_to_requested_size():
    collate_fn = _make_patch_collate_fn(patch_size=8)

    batch = [
        {
            "window": torch.rand(9, 16, 16),
            "mask": torch.ones(1, 16, 16),
            "histogram": torch.rand(64),
            "clean": torch.rand(3, 16, 16),
        },
        {
            "window": torch.rand(9, 16, 16),
            "mask": torch.ones(1, 16, 16),
            "histogram": torch.rand(64),
            "clean": torch.rand(3, 16, 16),
        },
    ]

    collated = collate_fn(batch)

    assert collated["window"].shape == (2, 9, 8, 8), f"Got {collated['window'].shape}"
    assert collated["mask"].shape == (2, 1, 8, 8)
    assert collated["clean"].shape == (2, 3, 8, 8)
    assert collated["histogram"].shape == (2, 64)


def test_patch_collate_fn_applies_same_offset_to_window_mask_clean():
    # window, mask, and clean must share the same random crop offset --
    # cropping them independently would misalign the training pair.
    torch.manual_seed(42)
    collate_fn = _make_patch_collate_fn(patch_size=8)

    # Encoding each position as i*16 + j lets the crop offset be recovered
    # from a single value after cropping, without tracking it separately.
    def make_position_encoded(shape):
        c, h, w = shape
        tensor = torch.zeros(c, h, w)
        for i in range(h):
            for j in range(w):
                tensor[:, i, j] = i * w + j
        return tensor

    window = make_position_encoded((9, 16, 16))
    mask = make_position_encoded((1, 16, 16))
    clean = make_position_encoded((3, 16, 16))

    batch = [
        {
            "window": window,
            "mask": mask,
            "histogram": torch.rand(64),
            "clean": clean,
        },
    ]

    collated = collate_fn(batch)

    cropped_window = collated["window"][0]
    cropped_mask = collated["mask"][0]
    cropped_clean = collated["clean"][0]

    # Inverts the i*16 + j encoding above: top=val//16, left=val%16.
    def infer_offset(cropped_tensor):
        first_val = cropped_tensor[0, 0, 0].item()
        top = int(first_val // 16)
        left = int(first_val % 16)
        return top, left

    window_offset = infer_offset(cropped_window)
    mask_offset = infer_offset(cropped_mask)
    clean_offset = infer_offset(cropped_clean)

    assert window_offset == mask_offset, \
        f"window offset {window_offset} != mask offset {mask_offset}"
    assert window_offset == clean_offset, \
        f"window offset {window_offset} != clean offset {clean_offset}"


def test_patch_collate_fn_clamps_to_actual_size_for_small_frames():
    # patch_size (256) exceeds the frame size (8x8) -- must clamp, not crash.
    collate_fn = _make_patch_collate_fn(patch_size=256)

    batch = [
        {
            "window": torch.rand(9, 8, 8),
            "mask": torch.ones(1, 8, 8),
            "histogram": torch.rand(64),
            "clean": torch.rand(3, 8, 8),
        },
    ]

    collated = collate_fn(batch)

    assert collated["window"].shape == (1, 9, 8, 8)
    assert collated["mask"].shape == (1, 1, 8, 8)
    assert collated["clean"].shape == (1, 3, 8, 8)


def test_train_one_epoch_with_patch_collate_fn():
    torch.manual_seed(0)
    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    collate_fn = _make_patch_collate_fn(patch_size=8)
    dataloader = torch.utils.data.DataLoader(
        _FixedBatchDataset(h=16, w=16, n=4),
        batch_size=4,
        collate_fn=collate_fn,
    )

    loss = train_one_epoch(model, optimizer, dataloader, device="cpu")
    assert loss == loss  # finite check (NaN != NaN)
    assert loss != float("inf")


def test_patch_collate_fn_handles_a_batch_of_differing_resolutions():
    # Source clips don't all share one resolution: preserving aspect ratio
    # when preparing them yields 854x480 alongside 854x450, and a batch can
    # mix the two. The collate used to call default_collate first, which
    # stacks -- so it raised "stack expects each tensor to be equal size"
    # before any cropping could equalise the shapes.
    collate_fn = _make_patch_collate_fn(patch_size=512)

    batch = [
        {
            "window": torch.rand(9, 480, 854),
            "mask": torch.ones(1, 480, 854),
            "histogram": torch.rand(64),
            "clean": torch.rand(3, 480, 854),
        },
        {
            "window": torch.rand(9, 450, 854),
            "mask": torch.ones(1, 450, 854),
            "histogram": torch.rand(64),
            "clean": torch.rand(3, 450, 854),
        },
    ]

    out = collate_fn(batch)

    # Everything crops to the smallest item's size, clamped by patch_size.
    assert out["window"].shape == (2, 9, 450, 512)
    assert out["mask"].shape == (2, 1, 450, 512)
    assert out["clean"].shape == (2, 3, 450, 512)
    assert out["histogram"].shape == (2, 64)


def test_patch_collate_fn_keeps_window_mask_clean_spatially_aligned():
    # A misaligned crop would pair a patch of the degraded window with a
    # different patch of the clean frame -- an invalid training label.
    collate_fn = _make_patch_collate_fn(patch_size=4)

    marker = torch.zeros(1, 16, 16)
    marker[0, 7, 9] = 1.0
    window = torch.zeros(9, 16, 16)
    window[0, 7, 9] = 1.0
    clean = torch.zeros(3, 16, 16)
    clean[0, 7, 9] = 1.0

    for _ in range(40):
        out = collate_fn([{"window": window, "mask": marker, "histogram": torch.rand(64), "clean": clean}])
        # Wherever the crop landed, the marker is either in all three or none.
        present = [bool(out[k][0, 0].max() > 0) for k in ("window", "mask", "clean")]
        assert len(set(present)) == 1


def test_patch_collate_crops_partner_tensors_with_the_same_offset():
    # The partner frame is the same clip one frame later, so it must be
    # cropped to the same window as the centre frame -- a different offset
    # would make the luminance delta between them a delta between two
    # different regions of the picture, which is not flicker.
    from training.train_mitigator import _make_patch_collate_fn

    collate = _make_patch_collate_fn(4)
    # A horizontal ramp: any misalignment between the two crops shows up as
    # a constant offset between them, which a uniform field could not reveal.
    ramp = torch.arange(8, dtype=torch.float32).repeat(8, 1)
    item = {
        "window": ramp.expand(9, 8, 8).clone(),
        "mask": ramp.expand(1, 8, 8).clone(),
        "clean": ramp.expand(3, 8, 8).clone(),
        "histogram": torch.zeros(64),
        "window_partner": ramp.expand(9, 8, 8).clone(),
        "mask_partner": ramp.expand(1, 8, 8).clone(),
        "clean_partner": ramp.expand(3, 8, 8).clone(),
        "histogram_partner": torch.zeros(64),
    }
    batch = collate([item, item])
    assert batch["window_partner"].shape == batch["window"].shape
    assert torch.equal(batch["window"][:, :3], batch["window_partner"][:, :3])
    assert torch.equal(batch["clean"], batch["clean_partner"])


class _FlickeringPairDataset(torch.utils.data.Dataset):
    """A pair whose clean frames are identical but whose degraded frames
    differ in brightness -- i.e. pure flicker, nothing else."""

    def __init__(self, h=16, w=16):
        torch.manual_seed(0)
        base = torch.rand(3, h, w) * 0.4 + 0.2
        self.clean = base
        self.mask = torch.ones(1, h, w)
        self.histogram = torch.rand(64)
        self.dark = torch.cat([base, base, base], dim=0)
        self.bright = torch.cat([base, (base + 0.4), (base + 0.4)], dim=0)

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {
            "window": self.dark, "mask": self.mask,
            "histogram": self.histogram, "clean": self.clean,
            "window_partner": self.bright, "mask_partner": self.mask,
            "histogram_partner": self.histogram, "clean_partner": self.clean,
        }


def test_lambda_temporal_zero_reproduces_the_reconstruction_only_objective():
    from training.train_mitigator import compute_losses

    model = MitigatorNet()
    batch = torch.utils.data.default_collate([_FlickeringPairDataset()[0]])
    losses = compute_losses(batch, model, device="cpu", lambda_temporal=0.0)
    assert losses["loss"].item() == pytest.approx(losses["reconstruction"].item(), abs=1e-6)


def test_temporal_term_charges_for_flicker_the_clean_pair_does_not_have():
    # Both clean frames are identical, so any brightness difference between
    # the two restored frames is pure residual flicker and must cost
    # something -- this is precisely what the per-frame reconstruction loss
    # cannot see.
    from training.train_mitigator import compute_losses

    model = MitigatorNet()
    batch = torch.utils.data.default_collate([_FlickeringPairDataset()[0]])
    losses = compute_losses(batch, model, device="cpu", lambda_temporal=1.0)
    assert losses["temporal"].item() > 0.0
    assert losses["loss"].item() > losses["reconstruction"].item()
