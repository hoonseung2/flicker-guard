"""Training loop for Mitigator: AdamW + L1/SSIM loss over
training.mitigator_dataset.MitigatorDataset, with checkpoint/resume support
so a multi-hour run surviving an interruption doesn't have to restart from
scratch (same philosophy as DatasetSynth's resumable batch CLI).
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, default_collate

from detector.profiles import load_profile
from mitigator.arch import MitigatorNet, mitigate_frame
from training.mitigator_dataset import MitigatorDataset
from training.mitigator_losses import l1_ssim_loss, psnr, temporal_consistency_loss


def save_checkpoint(
    path: Path,
    model: MitigatorNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float = float("inf"),
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(path: Path, model: MitigatorNet, optimizer: torch.optim.Optimizer) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    # Older checkpoints (written before best_val_loss was tracked) have no
    # such field -- fall back to inf rather than raising, since a checkpoint
    # missing the field is not a corrupted checkpoint, just an older one.
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    return checkpoint["epoch"], best_val_loss


def compute_losses(batch: dict, model: MitigatorNet, device: str, lambda_temporal: float) -> dict:
    """Restores the centre frame and its partner, and returns the loss terms.

    Both frames go through the model in a single forward pass (concatenated
    along the batch axis) rather than two calls: same arithmetic, one kernel
    launch sequence, and it keeps the reconstruction term applied to both
    outputs so the partner is never left graded only by the relative
    temporal term.
    """
    n = batch["window"].shape[0]
    window = torch.cat([batch["window"], batch["window_partner"]], dim=0).to(device)
    mask = torch.cat([batch["mask"], batch["mask_partner"]], dim=0).to(device)
    histogram = torch.cat([batch["histogram"], batch["histogram_partner"]], dim=0).to(device)
    clean = torch.cat([batch["clean"], batch["clean_partner"]], dim=0).to(device)

    restored = mitigate_frame(window, mask, histogram, model)
    reconstruction = l1_ssim_loss(restored, clean)
    temporal = temporal_consistency_loss(restored[:n], restored[n:], clean[:n], clean[n:])
    return {
        "loss": reconstruction + lambda_temporal * temporal,
        "reconstruction": reconstruction,
        "temporal": temporal,
        # PSNR stays a centre-frame-only measure so it remains comparable
        # with checkpoints trained before the partner frame existed.
        "psnr": psnr(restored[:n], clean[:n]),
    }


def train_one_epoch(
    model: MitigatorNet,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: str,
    lambda_temporal: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        optimizer.zero_grad()
        loss = compute_losses(batch, model, device, lambda_temporal)["loss"]
        loss_value = loss.item()
        if loss_value != loss_value:  # NaN check (NaN != NaN) -- fail loud, never train through it
            raise RuntimeError(
                "train_one_epoch: loss is NaN. This is a signal of a real bug "
                "(bad data, exploding learning rate, etc.), not something to "
                "silently train through -- backward() on a NaN loss would "
                "corrupt every weight in the model."
            )
        loss.backward()
        optimizer.step()

        total_loss += loss_value
        n_batches += 1
    return total_loss / n_batches


def evaluate(model: MitigatorNet, dataloader: DataLoader, device: str, lambda_temporal: float = 0.0) -> dict:
    model.eval()
    totals = {"loss": 0.0, "reconstruction": 0.0, "temporal": 0.0, "psnr": 0.0}
    n_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            losses = compute_losses(batch, model, device, lambda_temporal)
            for key in totals:
                totals[key] += losses[key].item()
            n_batches += 1
    return {key: value / n_batches for key, value in totals.items()}


def _make_patch_collate_fn(patch_size: int):
    """Creates a collate function that randomly crops all spatial tensors to patch_size.

    Window, mask, and clean are cropped with the same random offset to preserve their
    spatial alignment in the training pair (misaligned crops would produce invalid labels).
    Histogram (non-spatial vector) is left untouched. Crop size clamps to actual H/W if smaller.
    """
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")

    def collate_fn(batch_list):
        # Crop BEFORE stacking. Source clips don't all share one resolution --
        # preserving aspect ratio when preparing them yields e.g. 854x480
        # alongside 854x450 -- and default_collate stacks first, so calling it
        # up front raises "stack expects each tensor to be equal size" before
        # any cropping can equalise the shapes.
        #
        # The crop size is the smallest item's, not this item's: cropping each
        # to its own min(patch_size, h) would still leave a 450-tall and a
        # 480-tall crop in the same batch and fail at the stack just the same.
        crop_h = min(patch_size, min(item["window"].shape[-2] for item in batch_list))
        crop_w = min(patch_size, min(item["window"].shape[-1] for item in batch_list))

        cropped = []
        for item in batch_list:
            h, w = item["window"].shape[-2:]
            max_top = h - crop_h
            max_left = w - crop_w
            top = torch.randint(0, max_top + 1, (1,)).item() if max_top > 0 else 0
            left = torch.randint(0, max_left + 1, (1,)).item() if max_left > 0 else 0

            # Same (top, left) for every spatial tensor to keep them
            # aligned -- a misaligned crop would produce an invalid label,
            # and for the partner frame specifically it would turn the
            # temporal loss's luminance delta into a delta between two
            # different regions of the picture rather than a delta in time.
            crop = {}
            for key, value in item.items():
                crop[key] = (
                    value if value.dim() < 3
                    else value[:, top:top + crop_h, left:left + crop_w]
                )
            cropped.append(crop)

        return default_collate(cropped)

    return collate_fn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Mitigator on DatasetSynth output.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--profile", required=True, help="path to a profile JSON, e.g. configs/profiles/itu.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader worker processes. 0 (default) loads on the main thread, which "
             "starves the GPU once the dataset is large -- measured 12%% GPU utilisation "
             "and 596s/epoch on an H100. Set to roughly the core count.",
    )
    parser.add_argument(
        "--lambda-temporal", type=float, default=1.0,
        help="Weight on the temporal consistency term. L1/SSIM alone scores each "
             "frame independently, which a proportional correction satisfies while "
             "leaving the flicker pattern intact at lower amplitude -- measured on a "
             "real clip as a 60%% smaller brightness jump but only 12%% fewer "
             "transitions. 0 disables the term and reproduces the previous objective.",
    )
    parser.add_argument("--log-path")
    parser.add_argument("--patch-size", type=int, default=None,
                        help="Optional: random crop to this spatial size (e.g. 256) for VRAM-limited training on local GPUs. "
                             "Omit for full-resolution training on high-VRAM hardware (default: None, no crop).")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    profile = load_profile(Path(args.profile))
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path) if args.log_path else checkpoint_dir / "train_log.jsonl"

    train_dataset = MitigatorDataset(Path(args.data_dir), profile, split="train")
    val_dataset = MitigatorDataset(Path(args.data_dir), profile, split="val")
    print(
        f"train split: {len(train_dataset)} examples from "
        f"{train_dataset.samples_contributing}/{train_dataset.samples_scanned} contributing samples"
    )
    print(
        f"val split: {len(val_dataset)} examples from "
        f"{val_dataset.samples_contributing}/{val_dataset.samples_scanned} contributing samples"
    )
    if len(train_dataset) == 0:
        raise ValueError(
            "train split has 0 examples -- check --data-dir and --profile "
            "(no risky, prior-conditioned frame in any train-split sample)"
        )
    if len(val_dataset) == 0:
        raise ValueError(
            "val split has 0 examples -- check --data-dir and --profile "
            "(no risky, prior-conditioned frame in any val-split sample)"
        )

    # Create collate function if patch_size is specified. Validation is
    # deliberately excluded: a random crop makes every epoch's val loss
    # partly a function of crop luck rather than model quality, which would
    # make the best.pt comparison noisy -- val always runs at full
    # resolution, train-only VRAM limiting is what --patch-size is for.
    collate_fn = None
    if args.patch_size is not None:
        collate_fn = _make_patch_collate_fn(args.patch_size)

    # Each example reads four PNGs off disk, so with a single-threaded loader
    # the GPU starves: measured on an H100 with 9233 train examples, GPU
    # utilisation sat at 12% and an epoch took 596s. Workers overlap that
    # reading with compute. persistent_workers keeps them alive between
    # epochs, which matters here because there are only ~2300 batches per
    # epoch and re-forking each time would eat the gain.
    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs = {
            "num_workers": args.num_workers,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "pin_memory": device == "cuda",
        }

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, **loader_kwargs,
    )
    # Real source clips (e.g. DAVIS) don't all share one resolution, and val
    # is deliberately never cropped (see collate_fn=None above) to avoid
    # random-crop noise in the best.pt comparison -- so a val batch size > 1
    # can try to torch.stack differently-shaped frames from different clips
    # and crash. Validation doesn't need batching for throughput (no
    # backward pass), so batch_size=1 sidesteps this entirely while keeping
    # every val example at its real, uncropped resolution.
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=None, **loader_kwargs)

    model = MitigatorNet()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_val_loss = float("inf")
    latest_path = checkpoint_dir / "latest.pt"
    if args.resume and latest_path.exists():
        resumed_epoch, best_val_loss = load_checkpoint(latest_path, model, optimizer)
        start_epoch = resumed_epoch + 1

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        train_loss = train_one_epoch(model, optimizer, train_loader, device, args.lambda_temporal)
        val_metrics = evaluate(model, val_loader, device, args.lambda_temporal)
        elapsed = time.time() - start_time

        # best_val_loss must reflect this epoch's result before latest.pt is
        # saved, not after -- otherwise latest.pt gets the pre-update (stale,
        # one-epoch-behind) threshold, and a later --resume from latest.pt
        # can let a worse-than-true-best epoch pass the < check and clobber
        # best.pt with an inferior checkpoint.
        is_new_best = val_metrics["loss"] < best_val_loss
        if is_new_best:
            best_val_loss = val_metrics["loss"]

        save_checkpoint(latest_path, model, optimizer, epoch, best_val_loss)
        if is_new_best:
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, best_val_loss)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_reconstruction": val_metrics["reconstruction"],
                "val_temporal": val_metrics["temporal"],
                "val_psnr": val_metrics["psnr"],
                "lambda_temporal": args.lambda_temporal,
                "elapsed_seconds": elapsed,
            }) + "\n")
        # The two loss terms are printed separately because they trade off:
        # a run where val_temporal falls while val_reconstruction rises is
        # buying flicker suppression with picture fidelity, and the combined
        # number alone would hide which way that trade is going.
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
              f"(recon={val_metrics['reconstruction']:.4f} temporal={val_metrics['temporal']:.4f}) "
              f"val_psnr={val_metrics['psnr']:.2f} ({elapsed:.1f}s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
