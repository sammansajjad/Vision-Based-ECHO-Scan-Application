
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class GeometryTransform:
    """Record the geometry used to prepare an image for model input.

    All dimensions use (height, width) ordering.
    Padding values are measured in pixels on their named sides.
    """

    original_height: int
    original_width: int

    resized_height: int
    resized_width: int

    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int

    target_height: int
    target_width: int


def transform_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (256, 256),
) -> tuple[np.ndarray, GeometryTransform]:
    """Resize a 2D image proportionally and pad it to the target size."""

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    if image.ndim != 2:
        raise ValueError("image must be a 2D array")

    if image.size == 0:
        raise ValueError("image must not be empty")

    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("image must contain numeric values")

    if not np.isfinite(image).all():
        raise ValueError("image must contain only finite values")

    if (
        not isinstance(target_size, tuple)
        or len(target_size) != 2
        or any(
            isinstance(size, (bool, np.bool_))
            or not isinstance(size, (int, np.integer))
            or size <= 0
            for size in target_size
        )
    ):
        raise ValueError(
            "target_size must be a tuple of two positive integers "
            "(height, width)"
        )

    original_height, original_width = image.shape
    target_height, target_width = target_size

    # Fit the image inside the target dimensions without changing
    # its aspect ratio.
    scale = min(
        target_height / original_height,
        target_width / original_width,
    )

    resized_height = min(
        target_height,
        max(1, round(original_height * scale)),
    )
    resized_width = min(
        target_width,
        max(1, round(original_width * scale)),
    )

    # Center the resized image. An odd extra padding pixel goes
    # to the bottom or right.
    remaining_height = target_height - resized_height
    remaining_width = target_width - resized_width

    pad_top = remaining_height // 2
    pad_bottom = remaining_height - pad_top
    pad_left = remaining_width // 2
    pad_right = remaining_width - pad_left

    resized_image = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    transformed_image = cv2.copyMakeBorder(
        resized_image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    geometry = GeometryTransform(
        original_height=original_height,
        original_width=original_width,
        resized_height=resized_height,
        resized_width=resized_width,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
        target_height=target_height,
        target_width=target_width,
    )

    return transformed_image, geometry


def transform_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: tuple[int, int] = (256, 256),
) -> tuple[np.ndarray, np.ndarray, GeometryTransform]:
    """Transform an image and binary mask using identical geometry.

    Image interpolation is linear; mask interpolation is nearest-neighbor.
    Both are centered in the target dimensions using zero-padding.

    Returns:
        Transformed image, transformed uint8 binary mask, and geometry.
    """

    # Validate the image before accessing its shape.
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    if image.ndim != 2:
        raise ValueError("image must be a 2D array")

    if image.size == 0:
        raise ValueError("image must not be empty")

    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("image must contain numeric values")

    if not np.isfinite(image).all():
        raise ValueError("image must contain only finite values")

    # Validate the binary mask.
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a NumPy array")

    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    if mask.size == 0:
        raise ValueError("mask must not be empty")

    if not np.issubdtype(mask.dtype, np.number):
        raise TypeError("mask must contain numeric values")

    if not np.isfinite(mask).all():
        raise ValueError("mask must contain only finite values")

    if not np.isin(mask, [0, 1]).all():
        raise ValueError("mask must be binary, containing only 0 and 1")

    if image.shape != mask.shape:
        raise ValueError(
            "image and mask must have the same spatial dimensions"
        )

    # transform_image validates target_size and calculates the geometry.
    transformed_image, geometry = transform_image(image, target_size)

    # Use the exact same resized dimensions and padding for the mask.
    # Nearest-neighbor interpolation preserves discrete mask labels.
    resized_mask = cv2.resize(
        mask,
        (geometry.resized_width, geometry.resized_height),
        interpolation=cv2.INTER_NEAREST,
    )

    transformed_mask = cv2.copyMakeBorder(
        resized_mask,
        geometry.pad_top,
        geometry.pad_bottom,
        geometry.pad_left,
        geometry.pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    transformed_mask = transformed_mask.astype(np.uint8, copy=False)

    return transformed_image, transformed_mask, geometry
