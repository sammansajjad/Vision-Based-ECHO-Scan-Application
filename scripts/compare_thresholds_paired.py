
"""Paired per-frame threshold comparison on CAMUS validation.

Compares each threshold against 0.50 using the same frames.
Raw predictions only; no post-processing or training.
"""

from pathlib import Path

import numpy as np
import torch

from cv_pipeline.camus_dataset import CAMUSDataset
from cv_pipeline.models.unet import UNet


THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)
BASELINE = 0.50
TOLERANCE = 1e-12

CHECKPOINT_PATH = Path(
    "outputs/checkpoints/unet_4ch_best_full_validation.pth"
)

DEVICE = torch.device("cpu")


def dice_score(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    intersection = np.logical_and(gt, pred).sum()

    return float(
        (2.0 * intersection)
        / (gt.sum() + pred.sum() + 1e-8)
    )


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

    results = []

    print(f"Validation samples: {len(dataset)}")

    for i in range(len(dataset)):
        sample = dataset[i]

        x = torch.from_numpy(sample["image"])
        x = x.unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probability = (
                torch.sigmoid(model(x))
                .squeeze()
                .cpu()
                .numpy()
            )

        gt = sample["mask"]

        frame_result = {
            "patient_id": sample["patient_id"],
            "phase": sample["phase"],
            "dice": {},
        }

        for threshold in THRESHOLDS:
            pred = (probability >= threshold).astype(np.uint8)
            frame_result["dice"][threshold] = dice_score(gt, pred)

        results.append(frame_result)

        if (i + 1) % 20 == 0 or i == len(dataset) - 1:
            print(f"Evaluated {i + 1}/{len(dataset)}")

    print("\n=== PAIRED COMPARISON AGAINST THRESHOLD 0.50 ===")
    print("Improved = Dice increase | Worse = Dice decrease")

    for threshold in THRESHOLDS:
        if threshold == BASELINE:
            continue

        deltas = np.array([
            row["dice"][threshold] - row["dice"][BASELINE]
            for row in results
        ])

        improved = np.sum(deltas > TOLERANCE)
        worse = np.sum(deltas < -TOLERANCE)
        unchanged = len(deltas) - improved - worse

        print(f"\nThreshold: {threshold:.2f}")
        print(f"Improved:  {improved}")
        print(f"Worse:     {worse}")
        print(f"Unchanged: {unchanged}")
        print(f"Mean Dice change:   {deltas.mean():+.4f}")
        print(f"Median Dice change: {np.median(deltas):+.4f}")

    print("\n=== PAIRED COMPARISON BY PHASE ===")

    for phase in ("ED", "ES"):
        phase_rows = [
            row for row in results
            if row["phase"] == phase
        ]

        print(f"\n{phase} | Frames: {len(phase_rows)}")

        for threshold in THRESHOLDS:
            if threshold == BASELINE:
                continue

            deltas = np.array([
                row["dice"][threshold] - row["dice"][BASELINE]
                for row in phase_rows
            ])

            improved = np.sum(deltas > TOLERANCE)
            worse = np.sum(deltas < -TOLERANCE)
            unchanged = len(deltas) - improved - worse

            print(
                f"{threshold:.2f} | "
                f"Improved: {improved:2d} | "
                f"Worse: {worse:2d} | "
                f"Same: {unchanged:2d} | "
                f"Mean change: {deltas.mean():+.4f}"
            )


if __name__ == "__main__":
    main()
