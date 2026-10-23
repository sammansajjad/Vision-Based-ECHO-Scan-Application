
"""Initial training loop for binary CAMUS 4CH LV segmentation."""

import argparse
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cv_pipeline.camus_dataset import CAMUSDataset
from cv_pipeline.metrics import dice_coefficient, iou_score
from cv_pipeline.models.unet import UNet


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    val_metrics,
    seed,
):
    """Save model weights, optimizer state, and validation metrics."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "seed": seed,
        },
        path,
    )

    print(f"Checkpoint saved: {path}")


def make_loader(
    split: str,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a CAMUS DataLoader for 4CH ED and ES frames."""
    dataset = CAMUSDataset(
        split=split,
        view="4CH",
        phases=("ED", "ES"),
        target_size=(256, 256),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def evaluate(model, loader, device):
    """Evaluate sample-weighted loss and frame-level Dice and IoU."""
    model.eval()

    dice_scores = []
    iou_scores = []
    loss_total = 0.0
    sample_count = 0

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].unsqueeze(1).to(device)
            masks = batch["mask"].unsqueeze(1).float().to(device)

            logits = model(images)
            current_batch_size = images.shape[0]

            loss_total += (
                criterion(logits, masks).item() * current_batch_size
            )
            sample_count += current_batch_size

            predictions = (
                (torch.sigmoid(logits) >= 0.5)
                .cpu()
                .numpy()
            )
            truths = masks.cpu().numpy()

            for prediction, truth in zip(
                predictions[:, 0],
                truths[:, 0],
            ):
                prediction = prediction.astype(np.uint8)
                truth = truth.astype(np.uint8)

                dice_scores.append(dice_coefficient(prediction, truth))
                iou_scores.append(iou_score(prediction, truth))

    if sample_count == 0:
        raise ValueError("Cannot evaluate an empty DataLoader")

    return {
        "loss": loss_total / sample_count,
        "dice": float(np.mean(dice_scores)),
        "iou": float(np.mean(iou_scores)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train a U-Net for CAMUS 4CH LV segmentation."
    )

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=2)
    parser.add_argument("--max-val-batches", type=int, default=2)

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")

    if args.max_train_batches < 0 or args.max_val_batches < 0:
        parser.error("batch limits must be zero or positive")

    set_seed(args.seed)
    device = torch.device("cpu")

    train_loader = make_loader(
        "train",
        args.batch_size,
        shuffle=True,
    )
    val_loader = make_loader(
        "validation",
        args.batch_size,
        shuffle=False,
    )

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_path = os.path.join(
        "outputs",
        "checkpoints",
        "unet_4ch_latest.pth",
    )

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {1e-3}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Model parameters: {parameter_count}")
    print(f"Max train batches: {args.max_train_batches}")
    print(f"Max validation batches: {args.max_val_batches}")

    for epoch in range(args.epochs):
        model.train()
        batch_losses = []

        for batch_index, batch in enumerate(train_loader):
            if (
                args.max_train_batches > 0
                and batch_index >= args.max_train_batches
            ):
                break

            images = batch["image"].unsqueeze(1).to(device)
            masks = batch["mask"].unsqueeze(1).float().to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        if not batch_losses:
            raise ValueError("No training batches were processed.")

        if (
            args.max_val_batches > 0
            and args.max_val_batches < len(val_loader)
        ):
            validation_batches = []

            for index, batch in enumerate(val_loader):
                if index >= args.max_val_batches:
                    break
                validation_batches.append(batch)

            val_metrics = evaluate(
                model,
                validation_batches,
                device,
            )
        else:
            val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={np.mean(batch_losses):.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_dice={val_metrics['dice']:.4f} | "
            f"val_iou={val_metrics['iou']:.4f}"
        )

        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch + 1,
            val_metrics,
            args.seed,
        )


if __name__ == "__main__":
    main()
