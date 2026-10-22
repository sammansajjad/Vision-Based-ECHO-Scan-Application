import numpy as np


def _as_bool_mask(mask: np.ndarray, name: str) -> np.ndarray:
    """Validate a binary 2D mask (values 0/1 or bool) and return it as bool."""
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if mask.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if mask.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not (np.issubdtype(mask.dtype, np.number) or mask.dtype == np.bool_):
        raise TypeError(f"{name} must contain numeric or boolean values")
    if not np.isfinite(mask).all():
        raise ValueError(f"{name} must contain only finite values")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError(f"{name} must be binary, containing only 0 and 1")
    return mask.astype(bool)


def _overlap_counts(prediction: np.ndarray, ground_truth: np.ndarray):
    """Return (intersection, prediction_pixels, ground_truth_pixels)."""
    pred = _as_bool_mask(prediction, "prediction")
    truth = _as_bool_mask(ground_truth, "ground_truth")

    if pred.shape != truth.shape:
        raise ValueError(
            f"prediction and ground_truth shapes differ: "
            f"{pred.shape} vs {truth.shape}"
        )

    truth_pixels = int(truth.sum())
    if truth_pixels == 0:
        # A valid A4C frame contains the LV, so an empty ground truth is
        # bad data. Scoring it would hide the problem.
        raise ValueError("ground_truth has no foreground pixels")

    return int((pred & truth).sum()), int(pred.sum()), truth_pixels


def dice_coefficient(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Dice = 2|P and G| / (|P| + |G|). Empty prediction scores 0.0.

    Raises ValueError if ground_truth is empty.
    """
    intersection, pred_pixels, truth_pixels = _overlap_counts(
        prediction, ground_truth
    )
    return 2.0 * intersection / (pred_pixels + truth_pixels)


def iou_score(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """IoU = |P and G| / |P or G|. Empty prediction scores 0.0.

    Raises ValueError if ground_truth is empty.
    """
    intersection, pred_pixels, truth_pixels = _overlap_counts(
        prediction, ground_truth
    )
    return intersection / (pred_pixels + truth_pixels - intersection)