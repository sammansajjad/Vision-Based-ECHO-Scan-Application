"""Post-processing helpers for binary segmentation masks."""

import cv2
import numpy as np


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected region of a binary mask.

    Removes small false-positive specks far from the main structure.
    Returns a uint8 array with values 0 and 1. An empty mask is returned
    unchanged.
    """
    mask = np.asarray(mask).astype(np.uint8)

    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, received shape {mask.shape}")

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    # Label 0 is the background, so 1 label means the mask is empty.
    if num_labels <= 1:
        return mask

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_label).astype(np.uint8)