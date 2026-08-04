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
from training.mitigator_losses import l1_ssim_loss, psnr


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


def train_one_epoch(
    model: MitigatorNet,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        window = batch["window"].to(device)
        mask = batch["mask"].to(device)
        histogram = batch["histogram"].to(device)
        clean = batch["clean"].to(device)

        optimizer.zero_grad()
        restored = mitigate_frame(window, mask, histogram, model)
        loss = l1_ssim_loss(restored, clean)
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


def evaluate(model: MitigatorNet, dataloader: DataLoader, device: str) -> dict:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            window = batch["window"].to(device)
            mask = batch["mask"].to(device)
            histogram = batch["histogram"].to(device)
            clean = batch["clean"].to(device)

            restored = mitigate_frame(window, mask, histogram, model)
            total_loss += l1_ssim_loss(restored, clean).item()
            total_psnr += psnr(restored, clean).item()
            n_batches += 1
    return {"loss": total_loss / n_batches, "psnr": total_psnr / n_batches}


def _make_patch_collate_fn(patch_size: int):
    """Creates a collate function that randomly crops all spatial tensors to patch_size.

    Window, mask, and clean are cropped with the same random offset to preserve their
    spatial alignment in the training pair (misaligned crops would produce invalid labels).
    Histogram (non-spatial vector) is left untouched. Crop size clamps to actual H/W if smaller.
    """
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")

    def collate_fn(batch_list):
        batch_dict = default_collate(batch_list)

        window = batch_dict["window"]
        mask = batch_dict["mask"]
        clean = batch_dict["clean"]
        histogram = batch_dict["histogram"]

        batch_size, _, h, w = window.shape
        crop_h = min(patch_size, h)
        crop_w = min(patch_size, w)

        cropped_windows = []
        cropped_masks = []
        cropped_cleans = []

        for i in range(batch_size):
            max_top = max(0, h - crop_h)
            max_left = max(0, w - crop_w)
            top = torch.randint(0, max_top + 1, (1,)).item() if max_top > 0 else 0
            left = torch.randint(0, max_left + 1, (1,)).item() if max_left > 0 else 0

            # Use same (top, left) offset for window, mask, clean to keep them spatially aligned
            cropped_windows.append(window[i, :, top:top+crop_h, left:left+crop_w])
            cropped_masks.append(mask[i, :, top:top+crop_h, left:left+crop_w])
            cropped_cleans.append(clean[i, :, top:top+crop_h, left:left+crop_w])

        batch_dict["window"] = torch.stack(cropped_windows)
        batch_dict["mask"] = torch.stack(cropped_masks)
        batch_dict["clean"] = torch.stack(cropped_cleans)
        batch_dict["histogram"] = histogram

        return batch_dict

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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    # Real source clips (e.g. DAVIS) don't all share one resolution, and val
    # is deliberately never cropped (see collate_fn=None above) to avoid
    # random-crop noise in the best.pt comparison -- so a val batch size > 1
    # can try to torch.stack differently-shaped frames from different clips
    # and crash. Validation doesn't need batching for throughput (no
    # backward pass), so batch_size=1 sidesteps this entirely while keeping
    # every val example at its real, uncropped resolution.
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=None)

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
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
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
                "val_psnr": val_metrics["psnr"],
                "elapsed_seconds": elapsed,
            }) + "\n")
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
              f"val_psnr={val_metrics['psnr']:.2f} ({elapsed:.1f}s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
