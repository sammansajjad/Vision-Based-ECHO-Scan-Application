
"""EXP-011: Qualitative investigation of difficult LV segmentation cases.

Current case: patient0284, 4CH, ES.
Uses the official CAMUS validation split and existing U-Net checkpoint.

Predictions are raw binary masks at threshold 0.50.
No post-processing, retraining, or fine-tuning is performed.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from cv_pipeline.camus_adapter import load_camus_frame
from cv_pipeline.camus_dataset import CAMUSDataset
from cv_pipeline.mask_utils import to_binary_lv_mask
from cv_pipeline.metrics import dice_coefficient, iou_score
from cv_pipeline.model_input import transform_image_and_mask
from cv_pipeline.models.unet import UNet


# ---------------------------------------------------------
# 1. Experiment configuration
# ---------------------------------------------------------

PATIENT_ID = "patient0284"
VIEW = "4CH"
PHASE = "ES"

CHECKPOINT_PATH = Path(
    "outputs/checkpoints/unet_4ch_best_full_validation.pth"
)

OUTPUT_DIR = Path(
    "outputs/evaluations/exp011_failure_cases"
)

FIGURE_PATH = OUTPUT_DIR / (
    f"{PATIENT_ID}_{VIEW}_{PHASE}_failure_analysis.png"
)

# Patient-specific CSV prevents overwriting previous cases.
CSV_PATH = OUTPUT_DIR / (
    f"{PATIENT_ID}_{VIEW}_{PHASE}_failure_metrics.csv"
)

THRESHOLD = 0.50
DEVICE = torch.device("cpu")
TARGET_SIZE = (256, 256)


# ---------------------------------------------------------
# 2. Validate inputs and output paths
# ---------------------------------------------------------

if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

# Create the output directory if it does not exist.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Never overwrite an existing figure.
if FIGURE_PATH.exists():
    raise FileExistsError(
        f"Refusing to overwrite existing figure: {FIGURE_PATH}"
    )

# Never overwrite an existing CSV.
if CSV_PATH.exists():
    raise FileExistsError(
        f"Refusing to overwrite existing CSV: {CSV_PATH}"
    )


# ---------------------------------------------------------
# 3. Load the official validation sample
# ---------------------------------------------------------

dataset = CAMUSDataset(
    split="validation",
    view=VIEW,
    phases=("ED", "ES"),
)

if PATIENT_ID not in dataset.patient_ids:
    raise ValueError(
        f"{PATIENT_ID} is not in the official validation split."
    )

target_pair = (PATIENT_ID, PHASE)

if target_pair not in dataset.samples:
    raise ValueError(
        f"Sample not found in validation dataset: {target_pair}"
    )

sample_index = dataset.samples.index(target_pair)
sample = dataset[sample_index]

if sample["patient_id"] != PATIENT_ID:
    raise ValueError("Selected sample patient ID does not match.")

if sample["phase"] != PHASE:
    raise ValueError("Selected sample phase does not match.")

model_image = sample["image"]
ground_truth = sample["mask"]

if model_image.shape != TARGET_SIZE:
    raise ValueError(
        f"Unexpected model image shape: {model_image.shape}"
    )

if ground_truth.shape != TARGET_SIZE:
    raise ValueError(
        f"Unexpected ground-truth shape: {ground_truth.shape}"
    )


# ---------------------------------------------------------
# 4. Load original-resolution image and annotation
# ---------------------------------------------------------

image_input, original_labels, pixel_spacing = load_camus_frame(
    PATIENT_ID,
    view=VIEW,
    phase=PHASE,
)

original_image = image_input.image
original_mask = to_binary_lv_mask(original_labels)

if original_image.shape != original_mask.shape:
    raise ValueError(
        "Original image and mask shapes do not match."
    )

# Reproduce the dataset's exact image/mask transformation.
_, transformed_original_mask, original_geometry = (
    transform_image_and_mask(
        original_image,
        original_mask,
        target_size=TARGET_SIZE,
    )
)

if transformed_original_mask.shape != ground_truth.shape:
    raise ValueError(
        "Transformed original mask and dataset mask shapes differ."
    )

if not np.array_equal(transformed_original_mask, ground_truth):
    raise ValueError(
        "Original annotation transformation does not match "
        "the dataset ground-truth mask."
    )

# Convert GeometryTransform dataclass into a dictionary.
original_geometry_dict = {
    name: getattr(original_geometry, name)
    for name in original_geometry.__dataclass_fields__
}

# Compare actual geometry field values.
if original_geometry_dict != sample["geometry"]:
    raise ValueError(
        "Original and dataset geometry metadata do not match."
    )

print("Original-to-model mask correspondence: PASS")
print("Geometry metadata comparison: PASS")
print(f"Original image shape: {original_image.shape}")
print(f"Model input shape: {model_image.shape}")
print(f"Pixel spacing: {pixel_spacing}")


# ---------------------------------------------------------
# 5. Load the existing U-Net checkpoint
# ---------------------------------------------------------

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

model = UNet(
    in_channels=1,
    out_channels=1,
).to(DEVICE)

model.load_state_dict(state_dict)
model.eval()

print(f"Checkpoint loaded: {CHECKPOINT_PATH}")
print(f"Device: {DEVICE}")
print(f"Threshold: {THRESHOLD}")


# ---------------------------------------------------------
# 6. Generate raw prediction
# ---------------------------------------------------------

image_tensor = (
    torch.from_numpy(model_image)
    .unsqueeze(0)
    .unsqueeze(0)
    .to(DEVICE)
)

with torch.no_grad():
    logits = model(image_tensor)
    probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

# Raw prediction: no post-processing.
prediction = (
    probabilities >= THRESHOLD
).astype(np.uint8)

if prediction.shape != ground_truth.shape:
    raise ValueError(
        "Prediction and ground-truth shapes differ."
    )


# ---------------------------------------------------------
# 7. Calculate metrics and error counts
# ---------------------------------------------------------

gt_bool = ground_truth.astype(bool)
pred_bool = prediction.astype(bool)

true_positive = int(
    np.logical_and(pred_bool, gt_bool).sum()
)

false_positive = int(
    np.logical_and(pred_bool, ~gt_bool).sum()
)

false_negative = int(
    np.logical_and(~pred_bool, gt_bool).sum()
)

gt_pixels = int(gt_bool.sum())
pred_pixels = int(pred_bool.sum())

# Relative area difference requires a nonzero GT area.
if gt_pixels == 0:
    raise ValueError(
        "Ground-truth LV mask contains zero foreground pixels. "
        "Relative area difference cannot be calculated."
    )

dice = dice_coefficient(
    prediction,
    ground_truth,
)

iou = iou_score(
    prediction,
    ground_truth,
)

relative_area_difference = (
    (pred_pixels - gt_pixels) / gt_pixels * 100.0
)

print("\nEXP-011: Failure Case Analysis")
print("-" * 55)

print(f"Patient: {PATIENT_ID}")
print(f"View: {VIEW}")
print(f"Phase: {PHASE}")
print(f"Threshold: {THRESHOLD:.2f}")

print(f"\nGround-truth pixels: {gt_pixels}")
print(f"Predicted pixels: {pred_pixels}")

print(f"\nTrue positives: {true_positive}")
print(f"False positives: {false_positive}")
print(f"False negatives: {false_negative}")

print(f"\nDice: {dice:.6f}")
print(f"IoU: {iou:.6f}")

print(
    f"Relative area difference: "
    f"{relative_area_difference:.2f}%"
)


# ---------------------------------------------------------
# 8. Build TP / FP / FN error map
# ---------------------------------------------------------

# TP = green
# FP = red
# FN = blue

error_map = np.zeros(
    (*ground_truth.shape, 3),
    dtype=np.uint8,
)

error_map[
    np.logical_and(pred_bool, gt_bool)
] = (0, 200, 0)

error_map[
    np.logical_and(pred_bool, ~gt_bool)
] = (230, 0, 0)

error_map[
    np.logical_and(~pred_bool, gt_bool)
] = (0, 80, 255)


# ---------------------------------------------------------
# 9. Generate qualitative comparison figure
# ---------------------------------------------------------

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 10),
)

fig.suptitle(
    f"EXP-011 | {PATIENT_ID} | {VIEW} | {PHASE}\n"
    f"Raw U-Net prediction | "
    f"Threshold={THRESHOLD:.2f} | "
    f"Dice={dice:.4f} | IoU={iou:.4f}",
    fontsize=14,
)

# Original-resolution ultrasound.
axes[0, 0].imshow(
    original_image,
    cmap="gray",
)

axes[0, 0].set_title(
    f"Original ultrasound\n{original_image.shape}"
)

# Ground-truth LV mask.
axes[0, 1].imshow(
    ground_truth,
    cmap="gray",
    vmin=0,
    vmax=1,
)

axes[0, 1].set_title(
    "Ground-truth LV mask (256x256)"
)

# Raw prediction.
axes[0, 2].imshow(
    prediction,
    cmap="gray",
    vmin=0,
    vmax=1,
)

axes[0, 2].set_title(
    "Raw U-Net prediction (256x256)"
)

# Model input with ground-truth overlay.
axes[1, 0].imshow(
    model_image,
    cmap="gray",
)

axes[1, 0].imshow(
    ground_truth,
    cmap="Greens",
    alpha=0.35,
)

axes[1, 0].set_title(
    "Model input + GT overlay"
)

# Model input with prediction overlay.
axes[1, 1].imshow(
    model_image,
    cmap="gray",
)

axes[1, 1].imshow(
    prediction,
    cmap="Reds",
    alpha=0.35,
)

axes[1, 1].set_title(
    "Model input + prediction overlay"
)

# TP / FP / FN error map.
axes[1, 2].imshow(error_map)

axes[1, 2].set_title(
    "Error map: TP green | FP red | FN blue"
)

for axis in axes.flat:
    axis.axis("off")

fig.tight_layout()

fig.savefig(
    FIGURE_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)


# ---------------------------------------------------------
# 10. Export quantitative results
# ---------------------------------------------------------

fieldnames = [
    "patient_id",
    "view",
    "phase",
    "threshold",
    "ground_truth_pixels",
    "predicted_pixels",
    "true_positive",
    "false_positive",
    "false_negative",
    "dice",
    "iou",
    "relative_area_difference_percent",
    "original_height",
    "original_width",
    "model_height",
    "model_width",
]

# Recheck immediately before writing.
# This protects existing evidence if a file appeared meanwhile.
if CSV_PATH.exists():
    raise FileExistsError(
        f"Refusing to overwrite existing CSV: {CSV_PATH}"
    )

with CSV_PATH.open(
    "x",
    newline="",
    encoding="utf-8",
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerow({
        "patient_id": PATIENT_ID,
        "view": VIEW,
        "phase": PHASE,
        "threshold": THRESHOLD,
        "ground_truth_pixels": gt_pixels,
        "predicted_pixels": pred_pixels,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "dice": dice,
        "iou": iou,
        "relative_area_difference_percent": (
            relative_area_difference
        ),
        "original_height": original_image.shape[0],
        "original_width": original_image.shape[1],
        "model_height": model_image.shape[0],
        "model_width": model_image.shape[1],
    })


# ---------------------------------------------------------
# 11. Completion summary
# ---------------------------------------------------------

print("\n" + "=" * 55)
print("EXP-011 FAILURE CASE COMPLETED")
print("=" * 55)

print(f"Patient: {PATIENT_ID}")
print(f"View: {VIEW}")
print(f"Phase: {PHASE}")
print(f"Dice: {dice:.6f}")
print(f"IoU: {iou:.6f}")

print(f"\nFigure saved: {FIGURE_PATH}")
print(f"Metrics saved: {CSV_PATH}")

print("=" * 55)
