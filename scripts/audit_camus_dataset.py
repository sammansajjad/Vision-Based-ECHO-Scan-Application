
from pathlib import Path

import nibabel as nib
import numpy as np


CAMUS_ROOT = Path("data/raw/CAMUS_public/database_nifti")
SPLIT_ROOT = Path("data/raw/CAMUS_public/database_split")
SPLIT_FILES = (
    "subgroup_training.txt",
    "subgroup_validation.txt",
    "subgroup_testing.txt",
)
VIEWS = ("2CH", "4CH")
PHASES = ("ED", "ES")
EXPECTED_LABELS = {0, 1, 2, 3}


def load_split_patient_ids():
    """Load patient IDs from the official CAMUS split files."""
    split_ids = {}
    for filename in SPLIT_FILES:
        path = SPLIT_ROOT / filename
        if not path.is_file():
            raise FileNotFoundError(f"Split file not found: {path}")

        patient_ids = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if len(patient_ids) != len(set(patient_ids)):
            raise ValueError(f"Duplicate patient IDs in {path}")

        split_ids[filename] = patient_ids

    all_ids = [patient_id for ids in split_ids.values() for patient_id in ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Patient IDs overlap across official split files")

    return split_ids


def audit_pair(patient_id, view, phase):
    """Check one dedicated image and its matching ground-truth mask."""
    patient_dir = CAMUS_ROOT / patient_id
    image_path = patient_dir / f"{patient_id}_{view}_{phase}.nii.gz"
    mask_path = patient_dir / f"{patient_id}_{view}_{phase}_gt.nii.gz"
    errors = []

    if not image_path.is_file():
        errors.append(f"missing image: {image_path}")
    if not mask_path.is_file():
        errors.append(f"missing mask: {mask_path}")
    if errors:
        return errors

    try:
        image_nii = nib.load(str(image_path))
        mask_nii = nib.load(str(mask_path))

        image = np.asarray(image_nii.dataobj)
        mask = np.asarray(mask_nii.dataobj)

        if image.ndim != 2:
            errors.append(f"image is not 2D: shape={image.shape}")
        if mask.ndim != 2:
            errors.append(f"mask is not 2D: shape={mask.shape}")
        if image.shape != mask.shape:
            errors.append(
                f"shape mismatch: image={image.shape}, mask={mask.shape}"
            )
        if not np.allclose(image_nii.affine, mask_nii.affine):
            errors.append("image/mask affine mismatch")

        if mask.ndim == 2:
            labels = set(np.unique(mask).tolist())
            if not labels.issubset(EXPECTED_LABELS):
                errors.append(f"unexpected mask labels: {sorted(labels)}")

        if not np.isfinite(image).all():
            errors.append("image contains non-finite pixel values")

        if not np.isfinite(mask).all():
            errors.append("mask contains non-finite values")

    except Exception as exc:
        errors.append(f"load/check error: {type(exc).__name__}: {exc}")

    return errors


def main():
    split_ids = load_split_patient_ids()
    listed_ids = {
        patient_id
        for patient_ids in split_ids.values()
        for patient_id in patient_ids
    }
    disk_ids = {
        path.name
        for path in CAMUS_ROOT.glob("patient*")
        if path.is_dir()
    }

    print("CAMUS dataset integrity audit")
    print("=" * 32)

    for filename, patient_ids in split_ids.items():
        print(f"{filename}: {len(patient_ids)} patients")

    print(f"Unique split patients: {len(listed_ids)}")
    print(f"Patient directories: {len(disk_ids)}")
    print(f"Split IDs missing directories: {len(listed_ids - disk_ids)}")
    print(f"Directories absent from splits: {len(disk_ids - listed_ids)}")

    pair_count = 0
    failed_pairs = 0
    reported_errors = 0

    for patient_id in sorted(listed_ids):
        for view in VIEWS:
            for phase in PHASES:
                pair_count += 1
                errors = audit_pair(patient_id, view, phase)
                if errors:
                    failed_pairs += 1
                    for error in errors:
                        reported_errors += 1
                        print(f"[ERROR] {patient_id} {view} {phase}: {error}")

    print("\nAudit summary")
    print("=" * 32)
    print(f"Image-mask pairs checked: {pair_count}")
    print(f"Pairs with errors: {failed_pairs}")
    print(f"Reported errors: {reported_errors}")

    if listed_ids != disk_ids:
        print("RESULT: FAIL — split IDs and patient directories differ")
    elif failed_pairs:
        print("RESULT: FAIL — image-mask integrity errors found")
    else:
        print("RESULT: PASS — all checked pairs passed these integrity checks")


if __name__ == "__main__":
    main()
