import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from cv_pipeline.camus_adapter import load_camus_frame
from cv_pipeline.camus_splits import DEFAULT_SPLIT_ROOT, load_split_patient_ids
from cv_pipeline.mask_utils import to_binary_lv_mask
from cv_pipeline.model_input import transform_image_and_mask
from cv_pipeline.models.unet import UNet
from cv_pipeline.postprocess import keep_largest_component


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def dice_score(gt: np.ndarray, pr: np.ndarray) -> float:
    gt = gt.astype(bool)
    pr = pr.astype(bool)
    return float(2 * (gt & pr).sum() / (gt.sum() + pr.sum() + 1e-8))


def iou_score(gt: np.ndarray, pr: np.ndarray) -> float:
    gt = gt.astype(bool)
    pr = pr.astype(bool)
    return float((gt & pr).sum() / ((gt | pr).sum() + 1e-8))


def centroid(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def find_split(patient_id: str) -> str:
    split_ids = load_split_patient_ids(DEFAULT_SPLIT_ROOT)
    for split_name, ids in split_ids.items():
        if patient_id in ids:
            return split_name
    return "unknown"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

patient_id = sys.argv[1] if len(sys.argv) > 1 else "patient0003"
phase = sys.argv[2].upper() if len(sys.argv) > 2 else "ED"

checkpoint_path = Path("outputs/checkpoints/unet_4ch_best_full_validation.pth")

output_path = (
    Path("data/processed/figures")
    / f"{patient_id}_4CH_{phase}_unet_prediction.png"
)

device = torch.device("cpu")


# --------------------------------------------------
# 1. Load the original ECHO frame and its ground truth
# --------------------------------------------------

image_input, original_mask, spacing = load_camus_frame(patient_id, "4CH", phase)

original_image = np.asarray(image_input.image)
binary_mask = to_binary_lv_mask(original_mask)

print(f"Patient: {patient_id} (split: {find_split(patient_id)})")
print(f"View: 4CH | Phase: {phase}")
print(f"Original image shape: {original_image.shape}")
print(f"Original mask shape: {binary_mask.shape}")
print(f"Pixel spacing: {spacing}")


# --------------------------------------------------
# 2. Apply the SAME transformation used for training
# --------------------------------------------------

model_image_raw, model_mask, geometry = transform_image_and_mask(
    original_image,
    binary_mask,
    target_size=(256, 256),
)

# Same normalization as CAMUSDataset: float32 in [0, 1].
model_image = model_image_raw.astype(np.float32, copy=False) / 255.0

print(f"Model input shape: {model_image.shape}")
print(f"Ground-truth LV pixels: {int(model_mask.sum())}")


# --------------------------------------------------
# 3. Load the trained U-Net checkpoint
# --------------------------------------------------

if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

model = UNet(in_channels=1, out_channels=1).to(device)
model.load_state_dict(state_dict)
model.eval()


# --------------------------------------------------
# 4. Generate the LV segmentation prediction
# --------------------------------------------------

input_tensor = torch.from_numpy(model_image).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(input_tensor)
    probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

raw_prediction = (probabilities >= 0.5).astype(np.uint8)

# Remove small false-positive specks away from the LV.
prediction = keep_largest_component(raw_prediction)

dice_raw = dice_score(model_mask, raw_prediction)
dice = dice_score(model_mask, prediction)
iou = iou_score(model_mask, prediction)

gt_c = centroid(model_mask)
pr_c = centroid(prediction)

print(f"Predicted LV pixels (raw): {int(raw_prediction.sum())}")
print(f"Predicted LV pixels (largest component): {int(prediction.sum())}")
print(f"Dice raw: {dice_raw:.3f}")
print(f"Dice after post-processing: {dice:.3f} | IoU: {iou:.3f}")
print(
    f"Probability map: min={probabilities.min():.3f} "
    f"max={probabilities.max():.3f} mean={probabilities.mean():.3f}"
)

if gt_c is not None and pr_c is not None:
    dx, dy = pr_c[0] - gt_c[0], pr_c[1] - gt_c[1]
    print(
        f"Centroid GT=({gt_c[0]:.1f}, {gt_c[1]:.1f}) "
        f"Pred=({pr_c[0]:.1f}, {pr_c[1]:.1f}) "
        f"offset=(dx={dx:.1f}, dy={dy:.1f}) px in 256x256 space"
    )
else:
    print("Centroid: prediction or ground truth is empty")


# --------------------------------------------------
# 5. Visualize
# --------------------------------------------------

fig, axes = plt.subplots(1, 5, figsize=(24, 5))

# Model input image (same 256x256 space as all other panels)
axes[0].imshow(model_image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Model input (256x256)")

# Ground-truth overlay
axes[1].imshow(model_image, cmap="gray", vmin=0, vmax=1)
axes[1].imshow(
    np.ma.masked_where(model_mask == 0, model_mask),
    cmap="spring", vmin=0, vmax=1, alpha=0.65,
)
axes[1].set_title("Ground Truth Overlay")

# Prediction overlay
axes[2].imshow(model_image, cmap="gray", vmin=0, vmax=1)
axes[2].imshow(
    np.ma.masked_where(prediction == 0, prediction),
    cmap="spring", vmin=0, vmax=1, alpha=0.65,
)
axes[2].set_title("U-Net Prediction")

# GT vs prediction contours on the same panel
axes[3].imshow(model_image, cmap="gray", vmin=0, vmax=1)
if model_mask.any():
    axes[3].contour(model_mask.astype(float), levels=[0.5], colors="lime", linewidths=1.8)
if prediction.any():
    axes[3].contour(prediction.astype(float), levels=[0.5], colors="red", linewidths=1.8)
if gt_c is not None:
    axes[3].plot(*gt_c, "o", color="lime", markersize=6)
if pr_c is not None:
    axes[3].plot(*pr_c, "x", color="red", markersize=8, markeredgewidth=2)
axes[3].set_title(f"GT (green) vs Pred (red)\nDice {dice:.3f} | IoU {iou:.3f}")

# Probability heatmap
im = axes[4].imshow(probabilities, cmap="viridis", vmin=0, vmax=1)
axes[4].contour(model_mask.astype(float), levels=[0.5], colors="white", linewidths=1.0)
axes[4].set_title("Probability map (white = GT)")
fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

for ax in axes:
    ax.axis("off")

fig.suptitle(f"{patient_id} | 4CH | {phase} | U-Net LV Segmentation")
fig.tight_layout()

output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=120, bbox_inches="tight")
plt.close(fig)

print(f"\nVisualization saved to: {output_path}")