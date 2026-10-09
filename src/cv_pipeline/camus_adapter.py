
from pathlib import Path

import nibabel as nib
import numpy as np

from cv_pipeline.contracts import ImageInput


CAMUS_ROOT = Path("data/raw/CAMUS_public/database_nifti")


def load_camus_frame(
    patient_id: str,
    view: str = "4CH",
    phase: str = "ED",
) -> tuple[ImageInput, np.ndarray, tuple[float, ...]]:
    """Load one CAMUS frame and its matching ground-truth mask.

    Returns:
        image_input: Image and basic metadata in the shared project contract.
        mask: Original ground-truth labels; values are not binarized.
        pixel_spacing: Physical spacing reported by the image header.
    """
    view = view.upper()
    phase = phase.upper()

    if view not in {"2CH", "4CH"}:
        raise ValueError("view must be '2CH' or '4CH'")

    if phase not in {"ED", "ES"}:
        raise ValueError("phase must be 'ED' or 'ES'")

    patient_dir = CAMUS_ROOT / patient_id

    image_path = patient_dir / f"{patient_id}_{view}_{phase}.nii.gz"
    mask_path = patient_dir / f"{patient_id}_{view}_{phase}_gt.nii.gz"

    if not image_path.is_file():
        raise FileNotFoundError(f"CAMUS image not found: {image_path}")

    if not mask_path.is_file():
        raise FileNotFoundError(f"CAMUS mask not found: {mask_path}")

    image_nii = nib.load(str(image_path))
    mask_nii = nib.load(str(mask_path))

    image = np.asarray(image_nii.dataobj)
    mask = np.asarray(mask_nii.dataobj)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D image, received shape {image.shape}"
        )

    if mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D mask, received shape {mask.shape}"
        )

    if image.shape != mask.shape:
        raise ValueError(
            f"Image/mask shape mismatch: {image.shape} vs {mask.shape}"
        )

    if not np.allclose(image_nii.affine, mask_nii.affine):
        raise ValueError("Image and mask physical geometry do not match")

    spacing = tuple(float(value) for value in image_nii.header.get_zooms())

    image_input = ImageInput(
        image=image,
        patient_id=patient_id,
        modality="ECHO",
        view=view,
        frame_index=0,
    )

    return image_input, mask, spacing