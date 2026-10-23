"""Utilities for converting segmentation logits into binary masks."""

import torch


def logits_to_binary_masks(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Convert segmentation logits into binary masks.

    Args:
        logits: Raw model outputs, typically shaped [N, 1, H, W].
        threshold: Probability threshold in [0, 1].

    Returns:
        Binary torch.uint8 tensor with the same shape as logits.
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")

    if not logits.is_floating_point():
        raise TypeError("logits must have a floating-point dtype")

    if logits.ndim != 4:
        raise ValueError("logits must have shape [N, C, H, W]")

    if logits.numel() == 0:
        raise ValueError("logits must not be empty")

    if not torch.isfinite(logits).all().item():
        raise ValueError("logits must contain only finite values")

    if not isinstance(threshold, (int, float)) or isinstance(
        threshold, bool
    ):
        raise TypeError("threshold must be a number")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    probabilities = torch.sigmoid(logits)

    return (probabilities >= threshold).to(torch.uint8)