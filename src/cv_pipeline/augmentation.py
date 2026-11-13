"""Optional training-time augmentation for 2D image/mask pairs.

One random affine transform is applied jointly to the image and the mask
about the frame centre, followed by a multiplicative intensity gain on the
image only. There is deliberately no additive brightness offset, so zero
background and padding stay exactly zero. All randomness comes from the
NumPy Generator passed in by the caller; no global RNG is used.

The default ranges are untuned starting values, not tuned results.

Note on the foreground guard: it compares the augmented mask area with the
ORIGINAL mask area. Shrinking the image by a scale s also shrinks the mask
area by roughly s**2, so with the default 0.95 threshold any draw with
scale below about 0.975 is rejected even when the LV stays fully in frame.
Lower ``min_foreground_fraction`` if that is not wanted.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np


def _is_real_number(value) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(value)
    )


@dataclass(frozen=True)
class AugmentationConfig:
    """Ranges for the random augmentation (untuned defaults)."""

    max_rotation_deg: float = 10.0
    scale_range: tuple = (0.9, 1.1)
    # Fractions of the image size along each axis. Axis 1 is depth and has
    # tight LV margins, so its range is much smaller than axis 0.
    max_translation_axis0: float = 0.04
    max_translation_axis1: float = 0.01
    gain_range: tuple = (0.9, 1.1)
    # Minimum fraction of the original mask foreground that must survive.
    min_foreground_fraction: float = 0.95
    max_attempts: int = 5

    def __post_init__(self):
        for name in (
            "max_rotation_deg",
            "max_translation_axis0",
            "max_translation_axis1",
            "min_foreground_fraction",
        ):
            if not _is_real_number(getattr(self, name)):
                raise TypeError(f"{name} must be a finite number")

        for name in ("max_rotation_deg", "max_translation_axis0",
                     "max_translation_axis1"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

        for name in ("scale_range", "gain_range"):
            value = getattr(self, name)
            if (
                not isinstance(value, tuple)
                or len(value) != 2
                or not all(_is_real_number(v) for v in value)
            ):
                raise TypeError(f"{name} must be a tuple of two numbers")
            if value[0] <= 0 or value[0] > value[1]:
                raise ValueError(
                    f"{name} must satisfy 0 < low <= high"
                )

        if not 0.0 <= self.min_foreground_fraction <= 1.0:
            raise ValueError("min_foreground_fraction must be in [0, 1]")

        if (
            isinstance(self.max_attempts, (bool, np.bool_))
            or not isinstance(self.max_attempts, (int, np.integer))
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")


def _validate_inputs(image, mask, rng, config) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a NumPy array")
    if image.dtype != np.float32:
        raise TypeError("image must have dtype float32")
    if mask.dtype != np.uint8:
        raise TypeError("mask must have dtype uint8")
    if image.ndim != 2:
        raise ValueError("image must be a 2D array")
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if image.size == 0:
        raise ValueError("image must not be empty")
    if image.shape != mask.shape:
        raise ValueError("image and mask must have the same shape")
    if not np.isfinite(image).all():
        raise ValueError("image must contain only finite values")
    if image.min() < 0.0 or image.max() > 1.0:
        raise ValueError("image values must be in [0, 1]")
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError("mask must be binary, containing only 0 and 1")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if not isinstance(config, AugmentationConfig):
        raise TypeError("config must be an AugmentationConfig")


def augment_image_and_mask(image, mask, rng, config=None):
    """Return a jointly augmented copy of an image and its binary mask.

    Args:
        image: float32 2D array in [0, 1].
        mask: uint8 2D array containing only 0 and 1.
        rng: numpy.random.Generator; the only source of randomness.
        config: AugmentationConfig; defaults are used when omitted.

    Returns:
        (image_out, mask_out, params). Inputs are never modified and the
        outputs are always new arrays. If every attempt would push too much
        of the mask out of frame, unaugmented copies are returned with
        params {"applied": False, "attempts": max_attempts}.
    """
    if config is None:
        config = AugmentationConfig()

    _validate_inputs(image, mask, rng, config)

    height, width = image.shape
    centre = ((width - 1) / 2.0, (height - 1) / 2.0)

    original_foreground = int(mask.sum())
    minimum_foreground = config.min_foreground_fraction * original_foreground

    for attempt in range(1, int(config.max_attempts) + 1):
        angle = float(
            rng.uniform(-config.max_rotation_deg, config.max_rotation_deg)
        )
        scale = float(rng.uniform(*config.scale_range))
        shift0 = float(
            rng.uniform(
                -config.max_translation_axis0, config.max_translation_axis0
            )
        )
        shift1 = float(
            rng.uniform(
                -config.max_translation_axis1, config.max_translation_axis1
            )
        )
        gain = float(rng.uniform(*config.gain_range))

        # cv2 uses (x, y) = (axis 1, axis 0).
        matrix = cv2.getRotationMatrix2D(centre, angle, scale)
        matrix[0, 2] += shift1 * width
        matrix[1, 2] += shift0 * height

        mask_out = cv2.warpAffine(
            np.ascontiguousarray(mask),
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.uint8, copy=False)

        if original_foreground > 0 and int(mask_out.sum()) < (
            minimum_foreground
        ):
            continue

        image_out = cv2.warpAffine(
            np.ascontiguousarray(image),
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        image_out = np.clip(
            image_out * np.float32(gain), 0.0, 1.0
        ).astype(np.float32, copy=False)

        params = {
            "angle_deg": angle,
            "scale": scale,
            "translate_axis0": shift0,
            "translate_axis1": shift1,
            "gain": gain,
            "attempts": attempt,
            "applied": True,
        }
        return image_out, mask_out, params

    return (
        image.copy(),
        mask.copy(),
        {"applied": False, "attempts": int(config.max_attempts)},
    )