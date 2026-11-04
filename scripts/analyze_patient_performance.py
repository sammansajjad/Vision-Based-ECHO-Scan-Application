
"""EXP-010: Patient-level ED/ES segmentation performance analysis.

Evaluates the official CAMUS validation split:
- 4CH view
- ED and ES phases
- Raw LV predictions (no post-processing)
- Probability threshold: 0.50

Pairs frames by patient ID and exports one CSV row per patient.
"""

import csv
from pathlib import Path

import numpy as np
import torch

from cv_pipeline.camus_dataset import CAMUSDataset
from cv_pipeline.models.unet import UNet


# ---------------------------------------------------------
# 1. Experiment configuration
# ---------------------------------------------------------

CHECKPOINT_PATH = Path(
    "outputs/checkpoints/unet_4ch_best_full_validation.pth"
)

CSV_PATH = Path(
    "outputs/evaluations/exp010_patient_performance.csv"
)

EXPECTED_PATIENTS = 50
EXPECTED_FRAMES = 100

THRESHOLD = 0.50

DEVICE = torch.device("cpu")


# ---------------------------------------------------------
# 2. Dice metric
# ---------------------------------------------------------

def dice_score(ground_truth, prediction):
    """Calculate binary Dice score."""

    gt = ground_truth.astype(bool)
    pred = prediction.astype(bool)

    intersection = np.logical_and(gt, pred).sum()

    return float(
        (2.0 * intersection)
        / (gt.sum() + pred.sum() + 1e-8)
    )


# ---------------------------------------------------------
# 3. Check files and dataset integrity
# ---------------------------------------------------------

if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

if not CSV_PATH.parent.is_dir():
    raise FileNotFoundError(
        f"Evaluation directory not found: {CSV_PATH.parent}"
    )

if CSV_PATH.exists():
    raise FileExistsError(
        f"Refusing to overwrite existing CSV: {CSV_PATH}"
    )

dataset = CAMUSDataset(
    split="validation",
    view="4CH",
    phases=("ED", "ES"),
)

print("EXP-010: Patient-Level Performance Analysis")
print("-" * 55)

print(f"Dataset split: {dataset.split}")
print(f"View: {dataset.view}")
print(f"Phases: {dataset.phases}")
print(f"Dataset frames: {len(dataset)}")

# Check the official patient list.
dataset_patient_ids = list(dataset.patient_ids)

if len(dataset_patient_ids) != EXPECTED_PATIENTS:
    raise ValueError(
        f"Expected {EXPECTED_PATIENTS} validation patients, "
        f"found {len(dataset_patient_ids)}."
    )

if len(set(dataset_patient_ids)) != EXPECTED_PATIENTS:
    raise ValueError(
        "Duplicate patient IDs found in the validation patient list."
    )

if len(dataset) != EXPECTED_FRAMES:
    raise ValueError(
        f"Expected {EXPECTED_FRAMES} validation frames, "
        f"found {len(dataset)}."
    )

print("Initial dataset integrity checks: PASS")


# ---------------------------------------------------------
# 4. Load the existing trained model
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
print(f"Prediction threshold: {THRESHOLD}")


# ---------------------------------------------------------
# 5. Evaluate every validation frame
# ---------------------------------------------------------

# Structure:
# {
#     "patient0001": {"ED": 0.91, "ES": 0.82},
#     "patient0002": {"ED": 0.88, "ES": 0.90}
# }

patient_results = {}

seen_pairs = set()

phase_counts = {
    "ED": 0,
    "ES": 0,
}

for index in range(len(dataset)):

    sample = dataset[index]

    patient_id = sample["patient_id"]
    phase = sample["phase"]

    # Validate phase.
    if phase not in ("ED", "ES"):
        raise ValueError(
            f"Unexpected phase {phase!r} for {patient_id}."
        )

    # Detect duplicate patient-phase pairs.
    pair = (patient_id, phase)

    if pair in seen_pairs:
        raise ValueError(
            f"Duplicate patient-phase pair detected: {pair}"
        )

    seen_pairs.add(pair)

    # Prepare model input exactly as in the existing evaluator.
    image_tensor = (
        torch.from_numpy(sample["image"])
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    # Model inference.
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

    # Raw binary prediction: no post-processing.
    prediction = (
        probabilities >= THRESHOLD
    ).astype(np.uint8)

    ground_truth = sample["mask"]

    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch for {patient_id} {phase}: "
            f"prediction={prediction.shape}, "
            f"ground_truth={ground_truth.shape}"
        )

    frame_dice = dice_score(
        ground_truth,
        prediction,
    )

    # Store the result using patient ID and phase.
    if patient_id not in patient_results:
        patient_results[patient_id] = {}

    patient_results[patient_id][phase] = frame_dice

    phase_counts[phase] += 1

    if (index + 1) % 20 == 0 or index == len(dataset) - 1:
        print(f"Evaluated {index + 1}/{len(dataset)} frames")


# ---------------------------------------------------------
# 6. Validate patient-level pairing
# ---------------------------------------------------------

if len(seen_pairs) != EXPECTED_FRAMES:
    raise ValueError(
        f"Expected {EXPECTED_FRAMES} unique patient-phase pairs, "
        f"found {len(seen_pairs)}."
    )

if len(patient_results) != EXPECTED_PATIENTS:
    raise ValueError(
        f"Expected {EXPECTED_PATIENTS} unique patients, "
        f"found {len(patient_results)}."
    )

if phase_counts != {"ED": 50, "ES": 50}:
    raise ValueError(
        f"Unexpected phase counts: {phase_counts}"
    )

for patient_id, phases in patient_results.items():

    if set(phases.keys()) != {"ED", "ES"}:
        raise ValueError(
            f"Incomplete ED/ES pair for {patient_id}: {phases}"
        )

print("\nPatient pairing integrity: PASS")
print(f"Unique patients: {len(patient_results)}")
print(f"ED frames: {phase_counts['ED']}")
print(f"ES frames: {phase_counts['ES']}")
print(f"Duplicate pairs: 0")
print(f"Missing phases: 0")


# ---------------------------------------------------------
# 7. Calculate patient-level results
# ---------------------------------------------------------

patient_rows = []

for patient_id in sorted(patient_results):

    ed_dice = patient_results[patient_id]["ED"]
    es_dice = patient_results[patient_id]["ES"]

    mean_dice = (ed_dice + es_dice) / 2.0

    ed_minus_es = ed_dice - es_dice

    patient_rows.append({
        "patient_id": patient_id,
        "ED_dice": ed_dice,
        "ES_dice": es_dice,
        "mean_dice": mean_dice,
        "ED_minus_ES": ed_minus_es,
    })


# Convert patient-level measurements into arrays.
ed_scores = np.array([
    row["ED_dice"] for row in patient_rows
])

es_scores = np.array([
    row["ES_dice"] for row in patient_rows
])

mean_scores = np.array([
    row["mean_dice"] for row in patient_rows
])

differences = np.array([
    row["ED_minus_ES"] for row in patient_rows
])


# ---------------------------------------------------------
# 8. Overall descriptive statistics
# ---------------------------------------------------------

ed_higher = int(np.sum(differences > 0))
es_higher = int(np.sum(differences < 0))
ties = int(np.sum(differences == 0))

print("\n" + "=" * 55)
print("EXP-010 RESULTS")
print("=" * 55)

print("\nFrame-level phase means:")
print(f"ED mean Dice: {ed_scores.mean():.6f}")
print(f"ES mean Dice: {es_scores.mean():.6f}")

print("\nPatient-level mean Dice:")
print(f"Mean:   {mean_scores.mean():.6f}")
print(f"Median: {np.median(mean_scores):.6f}")
print(f"Std:    {mean_scores.std(ddof=0):.6f}")
print(f"Min:    {mean_scores.min():.6f}")
print(f"Max:    {mean_scores.max():.6f}")

print("\nED minus ES Dice difference:")
print(f"Mean difference:   {differences.mean():.6f}")
print(f"Median difference: {np.median(differences):.6f}")

print("\nPatient phase comparison:")
print(f"ED higher: {ed_higher}")
print(f"ES higher: {es_higher}")
print(f"Ties:      {ties}")


# ---------------------------------------------------------
# 9. Five lowest and highest performing patients
# ---------------------------------------------------------

sorted_patients = sorted(
    patient_rows,
    key=lambda row: row["mean_dice"],
)

print("\nFive lowest patient-level mean Dice:")

for row in sorted_patients[:5]:
    print(
        f"{row['patient_id']}: "
        f"ED={row['ED_dice']:.4f}, "
        f"ES={row['ES_dice']:.4f}, "
        f"Mean={row['mean_dice']:.4f}"
    )

print("\nFive highest patient-level mean Dice:")

for row in sorted_patients[-5:][::-1]:
    print(
        f"{row['patient_id']}: "
        f"ED={row['ED_dice']:.4f}, "
        f"ES={row['ES_dice']:.4f}, "
        f"Mean={row['mean_dice']:.4f}"
    )


# ---------------------------------------------------------
# 10. Export one row per patient
# ---------------------------------------------------------

fieldnames = [
    "patient_id",
    "ED_dice",
    "ES_dice",
    "mean_dice",
    "ED_minus_ES",
]

with CSV_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(patient_rows)

print("\n" + "=" * 55)
print("EXP-010 completed successfully.")
print(f"Patient records exported: {len(patient_rows)}")
print(f"CSV saved to: {CSV_PATH}")
print("=" * 55)
