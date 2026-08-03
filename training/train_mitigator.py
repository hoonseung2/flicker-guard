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
from torch.utils.data import DataLoader

from detector.profiles import load_profile
from mitigator.arch import MitigatorNet, mitigate_frame
from training.mitigator_dataset import MitigatorDataset
from training.mitigator_losses import l1_ssim_loss, psnr


def save_checkpoint(path: Path, model: MitigatorNet, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
        path,
    )


def load_checkpoint(path: Path, model: MitigatorNet, optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["epoch"]


def train_one_epoch(
    model: MitigatorNet,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: str,
) -> float:
    model.train()
    model.to(device)
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
    model.to(device)
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
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    profile = load_profile(Path(args.profile))
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path) if args.log_path else checkpoint_dir / "train_log.jsonl"

    train_dataset = MitigatorDataset(Path(args.data_dir), profile, split="train")
    val_dataset = MitigatorDataset(Path(args.data_dir), profile, split="val")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 0
    latest_path = checkpoint_dir / "latest.pt"
    if args.resume and latest_path.exists():
        start_epoch = load_checkpoint(latest_path, model, optimizer) + 1

    best_val_loss = float("inf")
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - start_time

        save_checkpoint(latest_path, model, optimizer, epoch)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch)

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
