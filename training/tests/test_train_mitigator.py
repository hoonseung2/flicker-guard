import contextlib
import io
import json
from pathlib import Path

import pytest
import torch

from detector.profiles import load_profile
from mitigator.arch import MitigatorNet
from training.train_mitigator import load_checkpoint, main, save_checkpoint, train_one_epoch, _make_patch_collate_fn

# Shared across tests that just need *a* profile and don't care which --
# real training always passes one explicitly via --profile.
_PROFILE = load_profile(Path("configs/profiles/itu.json"))


class _FixedBatchDataset(torch.utils.data.Dataset):
    """A dataset that always returns the same single example -- lets a
    training-loop test check "does loss go down" without needing real
    data/synthetic/ content or a GPU."""

    def __init__(self, h=16, w=16, n=4):
        torch.manual_seed(0)
        self.window = torch.rand(9, h, w)
        self.mask = torch.ones(1, h, w)
        self.strength = torch.rand(1)
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "window": self.window,
            "mask": self.mask,
            "strength": self.strength,
            # compute_batch_losses restores the centre frame and its temporal
            # partner together, so every example must carry both.
            "window_partner": self.window,
            "mask_partner": self.mask,
            "strength_partner": self.strength,
            # Required by compute_batch_losses's self-supervised loss --
            # zero shift, trusted, so the motion-dependent terms are live.
            "shift": torch.zeros(2),
            "shift_trusted": torch.ones(1),
        }


def test_train_one_epoch_reduces_loss_over_many_epochs():
    torch.manual_seed(0)
    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    dataloader = torch.utils.data.DataLoader(_FixedBatchDataset(), batch_size=4)

    first_epoch_loss = train_one_epoch(model, optimizer, dataloader, device="cpu", profile=_PROFILE)
    last_epoch_loss = first_epoch_loss
    for _ in range(19):
        last_epoch_loss = train_one_epoch(model, optimizer, dataloader, device="cpu", profile=_PROFILE)

    assert last_epoch_loss < first_epoch_loss


def test_train_one_epoch_returns_finite_loss():
    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataloader = torch.utils.data.DataLoader(_FixedBatchDataset(), batch_size=4)
    loss = train_one_epoch(model, optimizer, dataloader, device="cpu", profile=_PROFILE)
    assert loss == loss  # NaN check (NaN != NaN)
    assert loss != float("inf")


def test_checkpoint_save_and_load_restores_model_and_optimizer_state(tmp_path):
    torch.manual_seed(0)
    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataloader = torch.utils.data.DataLoader(_FixedBatchDataset(), batch_size=4)
    train_one_epoch(model, optimizer, dataloader, device="cpu", profile=_PROFILE)
    train_one_epoch(model, optimizer, dataloader, device="cpu", profile=_PROFILE)

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
    train_one_epoch(model, optimizer, dataloader, device="cpu", profile=_PROFILE)

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
    train_one_epoch(model, optimizer, dataloader, device="cuda", profile=_PROFILE)

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
    train_one_epoch(new_model, new_optimizer, dataloader, device="cuda", profile=_PROFILE)


class _FakeMitigatorDataset(torch.utils.data.Dataset):
    """Stands in for training.mitigator_dataset.MitigatorDataset in main()
    tests -- avoids needing a real data-dir with synthesized frames/priors
    on disk just to exercise main()'s checkpoint bookkeeping."""

    def __init__(self, *args, **kwargs):
        self.window = torch.rand(9, 8, 8)
        self.mask = torch.ones(1, 8, 8)
        self.strength = torch.rand(1)
        self.samples_scanned = 1
        self.samples_contributing = 1

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {
            "window": self.window,
            "mask": self.mask,
            "strength": self.strength,
            # compute_batch_losses restores the centre frame and its temporal
            # partner together, so every example must carry both.
            "window_partner": self.window,
            "mask_partner": self.mask,
            "strength_partner": self.strength,
            "shift": torch.zeros(2),
            "shift_trusted": torch.ones(1),
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

    def _fake_evaluate(model, dataloader, device, profile, lambda_temporal=1.0,
                        lambda_risk=1.0, tau=0.02, tau_area=0.05):
        return {
            "loss": next(scripted_losses), "fidelity": 0.0, "temporal": 0.0,
            "risk": 0.0, "soft_area": 0.0, "trusted_fraction": 1.0,
        }

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
            "strength": torch.rand(1),
            "window_partner": torch.rand(9, h, w),
            "mask_partner": torch.ones(1, h, w),
            "strength_partner": torch.rand(1),
            "shift": torch.zeros(2),
            "shift_trusted": torch.ones(1),
        }


def test_main_val_loader_handles_differently_shaped_examples(tmp_path, monkeypatch):
    # Regression test: real source clips (e.g. DAVIS) don't all share one
    # resolution, and val is deliberately never cropped -- a val batch size
    # > 1 with collate_fn=None used to try to torch.stack mismatched shapes
    # and crash with "stack expects each tensor to be equal size". main()
    # must hardcode val batch_size=1 to sidestep this.
    import training.train_mitigator as train_mitigator_module

    def _fake_dataset(data_dir, profile, split, **kwargs):
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

    def _nan_self_supervised_loss(out_a, out_b, in_a, in_b, dx, dy, profile,
                                   lambda_temporal=1.0, lambda_risk=1.0,
                                   tau=0.02, tau_area=0.05, weights=None):
        nan = torch.tensor(float("nan"))
        return {"loss": nan, "fidelity": nan, "temporal": nan, "risk": nan, "soft_area": nan}

    monkeypatch.setattr(train_mitigator_module, "self_supervised_loss", _nan_self_supervised_loss)

    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataloader = torch.utils.data.DataLoader(_FixedBatchDataset(), batch_size=4)

    with pytest.raises(RuntimeError, match="NaN"):
        train_one_epoch(model, optimizer, dataloader, device="cpu", profile=_PROFILE)


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

    loss = train_one_epoch(model, optimizer, dataloader, device="cpu", profile=_PROFILE)
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
    """A pair built from one shared base frame, degraded into a dark centre
    frame and a bright partner -- i.e. pure flicker, nothing else."""

    def __init__(self, h=16, w=16):
        torch.manual_seed(0)
        base = torch.rand(3, h, w) * 0.4 + 0.2
        self.mask = torch.ones(1, h, w)
        self.strength = torch.rand(1)
        self.dark = torch.cat([base, base, base], dim=0)
        self.bright = torch.cat([base, (base + 0.4), (base + 0.4)], dim=0)

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {
            "window": self.dark, "mask": self.mask,
            "strength": self.strength,
            "window_partner": self.bright, "mask_partner": self.mask,
            "strength_partner": self.strength,
            "shift": torch.zeros(2), "shift_trusted": torch.ones(1),
        }


def test_lambda_temporal_and_risk_zero_reduces_to_fidelity_only():
    # With both motion-dependent weights at 0, compute_batch_losses's "loss"
    # collapses to the fidelity term alone -- fidelity + 0 * temporal
    # + 0 * risk == fidelity.
    from training.train_mitigator import compute_batch_losses

    model = MitigatorNet()
    batch = torch.utils.data.default_collate([_FlickeringPairDataset()[0]])
    losses = compute_batch_losses(
        batch, model, device="cpu", profile=_PROFILE,
        lambda_temporal=0.0, lambda_risk=0.0,
    )
    assert losses["loss"].item() == pytest.approx(losses["fidelity"].item(), abs=1e-6)


def test_temporal_term_charges_for_flicker_the_partner_pair_has():
    # The degraded pair's centre frames differ only in brightness (pure
    # flicker), so the temporal term -- which compares the restored pair's
    # own frame-to-frame luminance delta, not a delta against a clean
    # reference -- must charge for it, and the combined loss must exceed
    # fidelity alone as a result.
    from training.train_mitigator import compute_batch_losses

    model = MitigatorNet()
    batch = torch.utils.data.default_collate([_FlickeringPairDataset()[0]])
    losses = compute_batch_losses(batch, model, device="cpu", profile=_PROFILE)
    assert losses["temporal"].item() > 0.0
    assert losses["loss"].item() > losses["fidelity"].item()


def test_patch_collate_fn_is_picklable():
    # Windows (and any spawn-based start method) pickles collate_fn to send
    # it to each DataLoader worker. A closure cannot be pickled, so a
    # closure-based collate raised "Can't get local object" at the first
    # batch -- invisible on Linux, which forks instead.
    import pickle

    from training.train_mitigator import _make_patch_collate_fn

    collate = _make_patch_collate_fn(8)
    restored = pickle.loads(pickle.dumps(collate))
    item = {
        "window": torch.rand(9, 16, 16), "mask": torch.ones(1, 16, 16),
        "clean": torch.rand(3, 16, 16), "histogram": torch.zeros(64),
        "window_partner": torch.rand(9, 16, 16), "mask_partner": torch.ones(1, 16, 16),
        "clean_partner": torch.rand(3, 16, 16), "histogram_partner": torch.zeros(64),
    }
    assert restored([item])["window"].shape == (1, 9, 8, 8)


class _SelfSupervisedBatchDataset(torch.utils.data.Dataset):
    """A flickering pair with everything the self-supervised loss needs."""

    def __init__(self, h=16, w=16, n=4):
        torch.manual_seed(0)
        dark = torch.rand(3, h, w) * 0.2
        bright = (dark + 0.7).clamp(0, 1)
        self.window = torch.cat([dark, dark, bright], dim=0)
        self.window_partner = torch.cat([dark, bright, bright], dim=0)
        self.mask = torch.ones(1, h, w)
        self.strength = torch.tensor([0.6])
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "window": self.window, "mask": self.mask, "strength": self.strength,
            "window_partner": self.window_partner, "mask_partner": self.mask,
            "strength_partner": self.strength,
            "shift": torch.zeros(2), "shift_trusted": torch.ones(1),
        }


def test_train_one_epoch_runs_on_the_self_supervised_loss():
    from training.train_mitigator import train_one_epoch

    torch.manual_seed(0)
    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(_SelfSupervisedBatchDataset(), batch_size=2)
    profile = load_profile(Path("configs/profiles/itu.json"))

    loss = train_one_epoch(model, optimizer, loader, device="cpu", profile=profile)
    assert loss == loss  # NaN check
    assert loss > 0.0


def test_evaluate_reports_every_loss_term():
    # The training log is how the fidelity/risk trade-off is watched. A
    # combined number alone hides which way it is going, and lambda_temporal
    # is the over-correction knob -- measured, not assumed.
    from training.train_mitigator import evaluate

    model = MitigatorNet()
    loader = torch.utils.data.DataLoader(_SelfSupervisedBatchDataset(), batch_size=1)
    profile = load_profile(Path("configs/profiles/itu.json"))

    metrics = evaluate(model, loader, device="cpu", profile=profile)
    for key in ("loss", "fidelity", "temporal", "risk", "soft_area", "trusted_fraction"):
        assert key in metrics


def test_train_one_epoch_is_silent_between_epochs_by_default():
    # The per-step line is opt-in: the epoch line is the record, and a
    # default-on step line would flood any existing log consumer.
    from training.train_mitigator import train_one_epoch

    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(_SelfSupervisedBatchDataset(), batch_size=2)
    profile = load_profile(Path("configs/profiles/itu.json"))

    with contextlib.redirect_stdout(io.StringIO()) as captured:
        train_one_epoch(model, optimizer, loader, device="cpu", profile=profile)

    assert captured.getvalue() == ""


def test_train_one_epoch_log_every_reports_each_term_and_flushes_the_last_window():
    # An epoch is 13+ minutes, so a 30-epoch run has no signal for hours
    # without this. Every term is reported separately for the same reason
    # the epoch line does it -- fidelity rising while temporal falls is the
    # over-correction signature, and the combined loss hides it.
    from training.train_mitigator import train_one_epoch

    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # 5 batches against log_every=2 leaves a partial final window: without
    # the n_batches == total_batches flush, step 5 would never be reported
    # and the epoch would appear to end at step 4.
    loader = torch.utils.data.DataLoader(_SelfSupervisedBatchDataset(n=5), batch_size=1)
    profile = load_profile(Path("configs/profiles/itu.json"))

    with contextlib.redirect_stdout(io.StringIO()) as captured:
        train_one_epoch(model, optimizer, loader, device="cpu", profile=profile, log_every=2)

    lines = [line for line in captured.getvalue().splitlines() if line.strip()]
    assert len(lines) == 3  # steps 2, 4, and the partial 5
    assert "step 5/5" in lines[-1]
    for line in lines:
        for term in ("loss=", "fidelity=", "temporal=", "risk=", "soft_area="):
            assert term in line


def test_an_untrusted_shift_drops_the_temporal_and_risk_terms():
    # A rejected phase-correlation estimate is indistinguishable from "no
    # motion" once it is a zero shift, so on a rejected pan both terms would
    # charge real camera movement as flicker. The example carries a trust
    # flag; the loop must honour it.
    from training.train_mitigator import compute_batch_losses

    model = MitigatorNet()
    profile = load_profile(Path("configs/profiles/itu.json"))
    batch = torch.utils.data.default_collate([_SelfSupervisedBatchDataset()[0]])
    batch["shift_trusted"] = torch.zeros(1, 1)

    terms = compute_batch_losses(batch, model, "cpu", profile)
    assert terms["temporal"].item() == pytest.approx(0.0, abs=1e-6)
    assert terms["risk"].item() == pytest.approx(0.0, abs=1e-6)
    assert terms["fidelity"].item() > 0.0 or terms["loss"].item() >= 0.0


class _DistinctPairDataset(torch.utils.data.Dataset):
    """A flickering pair whose content is seed-distinguishable from any
    other instance of this class -- lets a test tell whether one item's
    computed terms leaked information from a different item sharing its
    batch."""

    def __init__(self, seed, h=16, w=16):
        g = torch.Generator().manual_seed(seed)
        dark = torch.rand(3, h, w, generator=g) * 0.2
        bright = (dark + 0.7).clamp(0, 1)
        self.window = torch.cat([dark, dark, bright], dim=0)
        self.window_partner = torch.cat([dark, bright, bright], dim=0)
        self.mask = torch.ones(1, h, w)
        self.strength = torch.tensor([0.6])

    def item(self, shift_trusted: float) -> dict:
        return {
            "window": self.window, "mask": self.mask, "strength": self.strength,
            "window_partner": self.window_partner, "mask_partner": self.mask,
            "strength_partner": self.strength,
            "shift": torch.zeros(2), "shift_trusted": torch.tensor([shift_trusted]),
        }


def test_a_trusted_items_terms_do_not_depend_on_an_untrusted_batchmate():
    # Important-1 fix (code review, round 1): the trust gate used to be a
    # single batch-wide scalar (trusted.mean()) multiplied onto terms that
    # self_supervised_loss had ALREADY averaged across the batch. That both
    # let the untrusted item's raw value leak into the average at reduced
    # weight (never truly zeroed) and shrank the trusted item's own
    # contribution depending on how many untrusted items happened to share
    # its batch. The fix moves the gating inside self_supervised_loss, as a
    # per-item weight applied before the batch reduction -- this pins the
    # property that fix must deliver: a trusted item's temporal/risk in a
    # mixed batch must be bit-for-bit what it would be alone.
    #
    # MitigatorNet uses GroupNorm(1, ...), which normalises each sample
    # independently (no cross-sample statistics like BatchNorm), so the
    # model's own output for one item is unaffected by what else is in its
    # batch -- any difference here can only come from the loss reduction.
    from training.train_mitigator import compute_batch_losses

    model = MitigatorNet()
    profile = load_profile(Path("configs/profiles/itu.json"))

    trusted_item = _DistinctPairDataset(seed=1).item(shift_trusted=1.0)
    untrusted_item = _DistinctPairDataset(seed=2).item(shift_trusted=0.0)

    alone = compute_batch_losses(
        torch.utils.data.default_collate([trusted_item]), model, "cpu", profile,
    )
    mixed = compute_batch_losses(
        torch.utils.data.default_collate([trusted_item, untrusted_item]), model, "cpu", profile,
    )

    assert mixed["temporal"].item() == pytest.approx(alone["temporal"].item(), abs=1e-5)
    assert mixed["risk"].item() == pytest.approx(alone["risk"].item(), abs=1e-5)


def test_soft_area_matches_alone_and_trusted_fraction_reports_the_mix():
    # Important-2 fix (code review, round 1): soft_area was reported
    # ungated -- an operator reading val_soft_area could not tell how much
    # of it rested on rejected (mis-warped) pairs. Fixed the same way as
    # temporal/risk: gated per item inside self_supervised_loss, plus a new
    # trusted_fraction key so the operator can see how much of the batch
    # that number actually rests on.
    from training.train_mitigator import compute_batch_losses

    model = MitigatorNet()
    profile = load_profile(Path("configs/profiles/itu.json"))

    trusted_item = _DistinctPairDataset(seed=1).item(shift_trusted=1.0)
    untrusted_item = _DistinctPairDataset(seed=2).item(shift_trusted=0.0)

    alone = compute_batch_losses(
        torch.utils.data.default_collate([trusted_item]), model, "cpu", profile,
    )
    mixed = compute_batch_losses(
        torch.utils.data.default_collate([trusted_item, untrusted_item]), model, "cpu", profile,
    )

    assert mixed["soft_area"].item() == pytest.approx(alone["soft_area"].item(), abs=1e-5)
    assert mixed["trusted_fraction"].item() == pytest.approx(0.5, abs=1e-6)


def test_soft_area_and_trusted_fraction_are_zero_when_nothing_is_trusted():
    # Dividing by zero trusted examples must report 0.0, not NaN or a
    # division blow-up -- and trusted_fraction=0.0 is what tells the
    # operator that the 0.0 soft_area means "no usable data," not "no
    # flagged pixels."
    from training.train_mitigator import compute_batch_losses

    model = MitigatorNet()
    profile = load_profile(Path("configs/profiles/itu.json"))
    batch = torch.utils.data.default_collate([_SelfSupervisedBatchDataset()[0]])
    batch["shift_trusted"] = torch.zeros(1, 1)

    terms = compute_batch_losses(batch, model, "cpu", profile)
    assert terms["soft_area"].item() == pytest.approx(0.0, abs=1e-6)
    assert terms["trusted_fraction"].item() == pytest.approx(0.0, abs=1e-6)


def test_init_from_loads_weights_without_optimizer_or_epoch(tmp_path):
    from training.train_mitigator import load_model_weights

    torch.manual_seed(0)
    source = MitigatorNet()
    optimizer = torch.optim.AdamW(source.parameters(), lr=1e-2)
    dataloader = torch.utils.data.DataLoader(_FixedBatchDataset(), batch_size=4)
    train_one_epoch(source, optimizer, dataloader, device="cpu", profile=_PROFILE)
    path = tmp_path / "phase1.pt"
    save_checkpoint(path, source, optimizer, epoch=29, best_val_loss=0.17)

    target = MitigatorNet()
    load_model_weights(path, target)

    for a, b in zip(source.parameters(), target.parameters()):
        assert torch.equal(a, b)


def test_init_from_reports_an_architecture_mismatch_clearly(tmp_path):
    # in_channels went from 9+1+16 to 9+1+1 once already, and load_state_dict's
    # raw error names a tensor shape rather than the cause.
    from training.train_mitigator import load_model_weights

    path = tmp_path / "wrong.pt"
    torch.save({"model": {"nope": torch.zeros(1)}, "epoch": 0}, path)

    with pytest.raises(ValueError, match="in_channels"):
        load_model_weights(path, MitigatorNet())


def test_init_from_and_resume_together_is_an_error(tmp_path, monkeypatch):
    # --resume restores the epoch counter, so aiming it at a finished run
    # makes range(start_epoch, epochs) empty: zero epochs, exit code 0.
    import training.train_mitigator as train_module

    monkeypatch.setattr(train_module, "MitigatorDataset", _FakeMitigatorDataset)
    path = tmp_path / "phase1.pt"
    model = MitigatorNet()
    save_checkpoint(path, model, torch.optim.AdamW(model.parameters(), lr=1e-3), epoch=0)

    with pytest.raises(ValueError, match="mutually exclusive"):
        train_module.main([
            "--data-dir", str(tmp_path), "--profile", "configs/profiles/itu.json",
            "--checkpoint-dir", str(tmp_path / "ckpt"), "--epochs", "1",
            "--init-from", str(path), "--resume",
        ])


def test_split_file_is_passed_through_to_both_datasets(tmp_path, monkeypatch):
    import training.train_mitigator as train_module

    seen = []

    class _Recorder(_FakeMitigatorDataset):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("split_map"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(train_module, "MitigatorDataset", _Recorder)
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps({"a": "train", "b": "val"}), encoding="utf-8")

    train_module.main([
        "--data-dir", str(tmp_path), "--profile", "configs/profiles/itu.json",
        "--checkpoint-dir", str(tmp_path / "ckpt"), "--epochs", "1",
        "--split-file", str(split_file),
    ])

    assert seen == [{"a": "train", "b": "val"}, {"a": "train", "b": "val"}]


def test_snapshot_every_keeps_intermediate_epochs(tmp_path, monkeypatch):
    # Phase 1 kept only best.pt and latest.pt, so its lowest-soft_area epoch
    # (22 of 30) could not be evaluated afterwards at all.
    import training.train_mitigator as train_module

    monkeypatch.setattr(train_module, "MitigatorDataset", _FakeMitigatorDataset)
    checkpoint_dir = tmp_path / "ckpt"

    train_module.main([
        "--data-dir", str(tmp_path), "--profile", "configs/profiles/itu.json",
        "--checkpoint-dir", str(checkpoint_dir), "--epochs", "5",
        "--snapshot-every", "2",
    ])

    assert sorted(p.name for p in checkpoint_dir.glob("epoch_*.pt")) == [
        "epoch_000.pt", "epoch_002.pt", "epoch_004.pt"
    ]


def test_snapshots_are_off_by_default(tmp_path, monkeypatch):
    import training.train_mitigator as train_module

    monkeypatch.setattr(train_module, "MitigatorDataset", _FakeMitigatorDataset)
    checkpoint_dir = tmp_path / "ckpt"

    train_module.main([
        "--data-dir", str(tmp_path), "--profile", "configs/profiles/itu.json",
        "--checkpoint-dir", str(checkpoint_dir), "--epochs", "2",
    ])

    assert list(checkpoint_dir.glob("epoch_*.pt")) == []
