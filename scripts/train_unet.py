
"""Training loop for binary CAMUS 4CH LV segmentation."""

import argparse
import os
import random
import re

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
    validation_protocol,
    *,
    augment=False,
    run_name="",
):
    """Save model weights, optimizer state, and validation metadata."""
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
            "validation_protocol": validation_protocol,
            "augment": augment,
            "run_name": run_name,
        },
        path,
    )

    print(f"Checkpoint saved: {path}")


def load_best_dice(path, validation_protocol) -> float:
    """Load the best Dice score for the matching validation protocol."""
    if not os.path.exists(path):
        return -1.0

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        # Never compare scores from different validation protocols.
        if checkpoint.get("validation_protocol") != validation_protocol:
            print(
                "Warning: Existing best checkpoint uses a different "
                "validation protocol. It will not be used for comparison."
            )
            return -1.0

        metrics = checkpoint.get("val_metrics", {})
        dice = metrics.get("dice")

        if (
            isinstance(dice, (int, float))
            and np.isfinite(dice)
            and 0.0 <= dice <= 1.0
        ):
            return float(dice)

    except (OSError, RuntimeError, ValueError, TypeError, EOFError) as exc:
        print(
            f"Warning: Could not read best checkpoint ({exc}). "
            "It will not be used for comparison."
        )
        return -1.0

    print(
        "Warning: Existing best checkpoint has no valid Dice score. "
        "It will not be used for comparison."
    )
    return -1.0


def make_loader(
    split: str,
    batch_size: int,
    shuffle: bool,
    augment: bool = False,
) -> DataLoader:
    """Create a CAMUS DataLoader for 4CH ED and ES frames."""
    dataset = CAMUSDataset(
        split=split,
        view="4CH",
        phases=("ED", "ES"),
        target_size=(256, 256),
        augment=augment,
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

                dice_scores.append(
                    dice_coefficient(prediction, truth)
                )
                iou_scores.append(
                    iou_score(prediction, truth)
                )

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

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of frames per batch (default: 2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=2,
        help=(
            "Maximum training batches per epoch; use 0 for the full "
            "training split (default: 2, smoke test)."
        ),
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=2,
        help=(
            "Maximum validation batches; use 0 for the full validation "
            "split (default: 2, smoke test)."
        ),
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help=(
            "Enable training-only data augmentation (requires "
            "--run-name; default: off)."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help=(
            "Suffix for checkpoint file names so runs do not "
            "overwrite each other (letters, digits, - and _ only; "
            "default: none, which keeps the original file names)."
        ),
    )

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")

    if args.max_train_batches < 0 or args.max_val_batches < 0:
        parser.error("batch limits must be zero or positive")

    if args.run_name and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error(
            "--run-name may contain only letters, digits, '-' and '_'"
        )

    if args.augment and not args.run_name:
        parser.error("--augment requires --run-name")

    set_seed(args.seed)
    device = torch.device("cpu")

    train_loader = make_loader(
        "train",
        args.batch_size,
        shuffle=True,
        augment=args.augment,
    )
    val_loader = make_loader(
        "validation",
        args.batch_size,
        shuffle=False,
    )

    model = UNet().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_dir = os.path.join("outputs", "checkpoints")

    # An empty run name keeps the original checkpoint file names.
    run_suffix = f"_{args.run_name}" if args.run_name else ""

    latest_checkpoint_path = os.path.join(
        checkpoint_dir,
        f"unet_4ch_latest{run_suffix}.pth",
    )

    if args.max_val_batches == 0:
        validation_protocol = "full_validation"
    else:
        validation_protocol = (
            f"limited_validation_{args.max_val_batches}_batches"
        )

    best_checkpoint_path = os.path.join(
        checkpoint_dir,
        f"unet_4ch_best_{validation_protocol}{run_suffix}.pth",
    )

    best_dice = load_best_dice(
        best_checkpoint_path,
        validation_protocol,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Batch size: {args.batch_size}")
    print("Learning rate: 0.001")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Model parameters: {parameter_count}")
    print(f"Max train batches: {args.max_train_batches}")
    print(f"Max validation batches: {args.max_val_batches}")
    print(f"Validation protocol: {validation_protocol}")
    print(f"Augment: {args.augment}")
    print(f"Run name: {args.run_name or '(none)'}")
    print(f"Best validation Dice for this protocol: {best_dice:.4f}")

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
            val_metrics = evaluate(
                model,
                val_loader,
                device,
            )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={np.mean(batch_losses):.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_dice={val_metrics['dice']:.4f} | "
            f"val_iou={val_metrics['iou']:.4f}"
        )

        # Save the most recently completed epoch.
        save_checkpoint(
            latest_checkpoint_path,
            model,
            optimizer,
            epoch + 1,
            val_metrics,
            args.seed,
            validation_protocol,
            augment=args.augment,
            run_name=args.run_name,
        )

        # Preserve the best result for this validation protocol.
        current_dice = val_metrics["dice"]

        if current_dice > best_dice:
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                epoch + 1,
                val_metrics,
                args.seed,
                validation_protocol,
                augment=args.augment,
                run_name=args.run_name,
            )
            best_dice = current_dice
            print(f"New best validation Dice: {best_dice:.4f}")
        else:
            print(
                f"Best validation Dice remains: {best_dice:.4f}"
            )


if __name__ == "__main__":
    main()
