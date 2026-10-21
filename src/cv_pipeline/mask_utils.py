
import numpy as np


CAMUS_LABELS = {0, 1, 2, 3}
LV_CAVITY_LABEL = 1


def to_binary_lv_mask(mask: np.ndarray) -> np.ndarray:
    """Convert original CAMUS labels into a binary LV-cavity mask.

    Original labels:
        0: background
        1: LV cavity
        2: LV myocardium
        3: left atrium

    Returns:
        A uint8 array where 1 represents the LV cavity and 0 represents
        everything else. The input annotation is not modified.
    """
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, received shape {mask.shape}")

    if mask.size == 0:
        raise ValueError("Mask must not be empty")

    if not np.issubdtype(mask.dtype, np.number):
        raise ValueError("Mask must contain numeric labels")

    if not np.isfinite(mask).all():
        raise ValueError("Mask contains NaN or infinite values")

    observed_labels = set(np.unique(mask).tolist())
    unexpected_labels = observed_labels - CAMUS_LABELS

    if unexpected_labels:
        raise ValueError(
            f"Unexpected CAMUS labels: {sorted(unexpected_labels)}"
        )

    return (mask == LV_CAVITY_LABEL).astype(np.uint8)
