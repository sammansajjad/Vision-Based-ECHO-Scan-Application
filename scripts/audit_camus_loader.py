"""Integrity audit of the CAMUS 4CH loader, one split at a time.

Integrity audit only. If a problem is found in the test split, fix data
handling or document the exception. Do not change normalization, target
size or any other design choice based on what test frames look like.

Reads data only: no model, no scoring, no files written.

Usage:
    python scripts\\audit_camus_loader.py            (train)
    python scripts\\audit_camus_loader.py validation
    python scripts\\audit_camus_loader.py test
"""

import sys
from collections import defaultdict

import numpy as np

from cv_pipeline.camus_dataset import CAMUSDataset

VALID_SPLITS = ("train", "validation", "test")
EXPECTED_SHAPE = (256, 256)


def audit_split(split):
    dataset = CAMUSDataset(split=split, view="4CH", phases=("ED", "ES"))

    print(f"Split: {split}")
    print(f"Patients: {len(dataset.patient_ids)}")
    print(f"Enumerated samples: {len(dataset)}")

    loaded = 0
    failures = []
    problems = []  # loaded samples that violate the output contract

    phases_loaded = defaultdict(set)
    source_heights, source_widths = [], []
    image_min, image_max = np.inf, -np.inf
    mask_values = set()
    lv_pixels = []
    pad_nonzero = 0

    for index, (patient_id, phase) in enumerate(dataset.samples):
        try:
            sample = dataset[index]
        except Exception as exc:  # audit must collect every failure
            failures.append(
                (patient_id, phase, f"{type(exc).__name__}: {exc}")
            )
            continue

        loaded += 1
        phases_loaded[patient_id].add(phase)

        image, mask, geo = sample["image"], sample["mask"], sample["geometry"]
        label = f"{patient_id} {phase}"

        if image.shape != EXPECTED_SHAPE or mask.shape != EXPECTED_SHAPE:
            problems.append(
                f"{label}: shapes {image.shape}/{mask.shape}"
            )
        if not np.isfinite(image).all():
            problems.append(f"{label}: non-finite image values")

        image_min = min(image_min, float(image.min()))
        image_max = max(image_max, float(image.max()))
        mask_values |= {int(v) for v in np.unique(mask)}

        lv_count = int(mask.sum())
        lv_pixels.append(lv_count)
        if lv_count == 0:
            problems.append(f"{label}: empty LV mask")

        source_heights.append(geo["original_height"])
        source_widths.append(geo["original_width"])

        # Padding must be zero in both image and mask.
        top, bottom = geo["pad_top"], geo["pad_bottom"]
        left, right = geo["pad_left"], geo["pad_right"]
        h, w = image.shape
        for array in (image, mask):
            border = [
                array[:top, :], array[h - bottom:, :],
                array[:, :left], array[:, w - right:],
            ]
            if any(part.size and np.any(part != 0) for part in border):
                pad_nonzero += 1
                problems.append(f"{label}: non-zero padding")
                break

    both_phases = sum(1 for p in phases_loaded.values() if p == {"ED", "ES"})

    print(f"Loaded successfully: {loaded}")
    print(f"Failed: {len(failures)}")
    for patient_id, phase, message in failures:
        print(f"  FAIL {patient_id} {phase}: {message}")

    if loaded:
        print(
            f"Source height range: {min(source_heights)}-{max(source_heights)}"
            f" | width range: {min(source_widths)}-{max(source_widths)}"
        )
        print(f"Mask unique values (all samples): {sorted(mask_values)}")
        print(
            f"LV pixels per mask: min {min(lv_pixels)}, "
            f"median {int(np.median(lv_pixels))}, max {max(lv_pixels)}"
        )
        print(f"Scaled image range: [{image_min:.4f}, {image_max:.4f}]")

    print(
        f"Patients with both ED and ES loaded: {both_phases}"
        f" of {len(dataset.patient_ids)}"
    )
    print(f"Contract problems: {len(problems)}")
    for message in problems:
        print(f"  PROBLEM {message}")

    return len(failures) + len(problems) + (
        0 if both_phases == len(dataset.patient_ids) else 1
    )


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    if split not in VALID_SPLITS:
        print(f"Unknown split {split!r}; expected one of {VALID_SPLITS}")
        return 2

    issues = audit_split(split)
    print("RESULT:", "PASS" if issues == 0 else "FAIL")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())