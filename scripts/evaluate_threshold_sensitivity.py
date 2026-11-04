
"""Compare raw U-Net segmentation metrics at different thresholds.

Validation-only experiment. Does not train or modify the model.
"""

from pathlib import Path

import numpy as np
import torch

from cv_pipeline.camus_dataset import CAMUSDataset
from cv_pipeline.models.unet import UNet


THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)

CHECKPOINT_PATH = Path(
    "outputs/checkpoints/unet_4ch_best_full_validation.pth"
)

DEVICE = torch.device("cpu")


def calculate_metrics(gt, pred):
    """Calculate per-frame Dice and IoU."""
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()

    dice = (2.0 * intersection) / (gt.sum() + pred.sum() + 1e-8)
    iou = intersection / (union + 1e-8)

    return float(dice), float(iou)


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    dataset = CAMUSDataset(
        split="validation",
        view="4CH",
        phases=("ED", "ES"),
    )

    results = {t: [] for t in THRESHOLDS}

    print(f"Validation samples: {len(dataset)}")

    for i in range(len(dataset)):
        sample = dataset[i]

        x = torch.from_numpy(sample["image"])
        x = x.unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probability = torch.sigmoid(model(x)).squeeze().cpu().numpy()

        gt = sample["mask"]

        for threshold in THRESHOLDS:
            prediction = (probability >= threshold).astype(np.uint8)

            dice, iou = calculate_metrics(gt, prediction)

            results[threshold].append({
                "dice": dice,
                "iou": iou,
                "patient_id": sample["patient_id"],
                "phase": sample["phase"],
            })

        if (i + 1) % 20 == 0 or i == len(dataset) - 1:
            print(f"Evaluated {i + 1}/{len(dataset)}")

    print("\n=== VALIDATION THRESHOLD COMPARISON ===")
    print("Raw predictions | No post-processing")
    print(f"{'Threshold':<12}{'Mean Dice':<14}{'Mean IoU':<14}")

    for threshold in THRESHOLDS:
        rows = results[threshold]

        mean_dice = np.mean([r["dice"] for r in rows])
        mean_iou = np.mean([r["iou"] for r in rows])

        print(f"{threshold:<12.2f}{mean_dice:<14.4f}{mean_iou:<14.4f}")

    print("\n=== ED / ES DICE COMPARISON ===")

    for threshold in THRESHOLDS:
        rows = results[threshold]

        for phase in ("ED", "ES"):
            values = [
                r["dice"] for r in rows
                if r["phase"] == phase
            ]

            print(
                f"Threshold {threshold:.2f} | "
                f"{phase}: {np.mean(values):.4f}"
            )

    print("\n=== PATIENT0291 ES ===")

    for threshold in THRESHOLDS:
        matching = [
            r for r in results[threshold]
            if r["patient_id"] == "patient0291"
            and r["phase"] == "ES"
        ]

        if matching:
            row = matching[0]
            print(
                f"Threshold {threshold:.2f} | "
                f"Dice: {row['dice']:.4f} | "
                f"IoU: {row['iou']:.4f}"
            )


if __name__ == "__main__":
    main()
