
"""Evaluate a saved U-Net checkpoint and visualize one CAMUS frame."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from cv_pipeline.camus_dataset import CAMUSDataset
from cv_pipeline.metrics import dice_coefficient, iou_score
from cv_pipeline.models.unet import UNet
from cv_pipeline.prediction import logits_to_binary_masks


def main():
    # 1. Load one real 4CH end-diastolic validation sample.
    dataset = CAMUSDataset(
        split="validation",
        view="4CH",
        phases=("ED",),
    )
    sample = dataset[0]
    print("Patient:", sample["patient_id"])
    print("View:", sample["view"])
    print("Phase:", sample["phase"])
    print("Image shape:", sample["image"].shape)
    print("Mask shape:", sample["mask"].shape)
    print("Image min/max:", sample["image"].min(), sample["image"].max())
    print("Mask unique values:", set(sample["mask"].flatten().tolist()))
    print("Ground-truth LV pixels:", int(sample["mask"].sum()))

    # 2. Restore the saved U-Net checkpoint.
    checkpoint_path = Path(
        "outputs/checkpoints/unet_4ch_latest.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model = UNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Prepare the image for the model.
    # Dataset image: [H, W]
    # Model input: [batch, channels, H, W]
    image = torch.from_numpy(sample["image"]).unsqueeze(0).unsqueeze(0)
    ground_truth = sample["mask"]

    # 4. Generate a prediction without calculating gradients.
    with torch.no_grad():
        logits = model(image)
        prediction = logits_to_binary_masks(
            logits,
            threshold=0.5,
        )[0, 0].numpy()

    # 5. Calculate segmentation metrics.
    dice = dice_coefficient(prediction, ground_truth)
    iou = iou_score(prediction, ground_truth)

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Patient:", sample["patient_id"])
    print("View:", sample["view"])
    print("Phase:", sample["phase"])
    print("Image shape:", sample["image"].shape)
    print("Prediction shape:", prediction.shape)
    print("Predicted LV pixels:", int(prediction.sum()))
    print("Ground-truth LV pixels:", int(ground_truth.sum()))
    print(f"Dice: {dice:.4f}")
    print(f"IoU: {iou:.4f}")

    # 6. Display the image and both segmentation masks.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(sample["image"], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original ECHO Image")

    axes[1].imshow(sample["image"], cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(
        ground_truth,
        cmap="Blues",
        alpha=0.55,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Ground Truth LV Mask")

    axes[2].imshow(sample["image"], cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(
        prediction,
        cmap="Reds",
        alpha=0.55,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Predicted LV Mask")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"{sample['patient_id']} | {sample['view']} | {sample['phase']}"
        f"\nDice={dice:.4f} | IoU={iou:.4f}"
    )
    fig.tight_layout()

    # 7. Save the visualization.
    output_path = Path("outputs/evaluations")
    output_path.mkdir(parents=True, exist_ok=True)

    image_path = output_path / "patient0200_4CH_ED_comparison.png"
    fig.savefig(image_path, dpi=150, bbox_inches="tight")
    print("Visualization saved:", image_path)

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
