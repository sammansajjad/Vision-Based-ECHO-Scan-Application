
from typing import Callable

import numpy as np
from torch.utils.data import Dataset

from cv_pipeline.augmentation import (
    AugmentationConfig,
    augment_image_and_mask,
)
from cv_pipeline.camus_adapter import load_camus_frame
from cv_pipeline.camus_splits import (
    DEFAULT_SPLIT_ROOT,
    load_split_patient_ids,
)
from cv_pipeline.mask_utils import to_binary_lv_mask
from cv_pipeline.model_input import transform_image_and_mask


FrameLoader = Callable[
    [str, str, str],
    tuple[object, np.ndarray, tuple[float, ...]],
]


class CAMUSDataset(Dataset):
    """On-demand CAMUS dataset for binary LV-cavity segmentation.

    Each sample represents one patient, one view, and one cardiac phase.
    Patient splits come from the official CAMUS split files.

    Images are returned as float32 arrays in [0, 1], with shape (H, W).
    Masks are returned as uint8 binary arrays, with shape (H, W).

    This class prepares samples only; it does not train a model.
    """

    VALID_SPLITS = ("train", "validation", "test")
    VALID_PHASES = ("ED", "ES")
    VALID_VIEWS = ("2CH", "4CH")

    def __init__(
        self,
        split: str = "train",
        view: str = "4CH",
        phases: tuple[str, ...] = ("ED", "ES"),
        target_size: tuple[int, int] = (256, 256),
        split_root=DEFAULT_SPLIT_ROOT,
        frame_loader: FrameLoader = load_camus_frame,
        augment: bool = False,
        augment_seed: int = 0,
    ):
        split = split.lower()
        view = view.upper()
        phases = tuple(phase.upper() for phase in phases)

        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Unknown split {split!r}; expected one of "
                f"{self.VALID_SPLITS}"
            )

        if view not in self.VALID_VIEWS:
            raise ValueError("view must be '2CH' or '4CH'")

        if not phases:
            raise ValueError("phases must contain at least one phase")

        if len(set(phases)) != len(phases):
            raise ValueError("phases must not contain duplicates")

        if any(phase not in self.VALID_PHASES for phase in phases):
            raise ValueError("phases must contain only 'ED' and/or 'ES'")

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
                "target_size must be a tuple of two positive integers"
            )

        if not isinstance(augment, bool):
            raise TypeError("augment must be a bool")

        if isinstance(augment_seed, (bool, np.bool_)) or not isinstance(
            augment_seed, (int, np.integer)
        ):
            raise TypeError("augment_seed must be an int")

        if augment_seed < 0:
            raise ValueError("augment_seed must be non-negative")

        self.split = split
        self.view = view
        self.phases = phases
        self.target_size = tuple(int(size) for size in target_size)
        self.frame_loader = frame_loader
        self.augment = augment
        self.augment_seed = int(augment_seed)

        if self.augment:
            self._augment_config = AugmentationConfig()
            self._rng = np.random.default_rng(self.augment_seed)

        split_ids = load_split_patient_ids(split_root)
        self.patient_ids = tuple(split_ids[split])

        self.samples = tuple(
            (patient_id, phase)
            for patient_id in self.patient_ids
            for phase in self.phases
        )

    def __len__(self) -> int:
        """Return the number of patient-phase samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        """Load and prepare one sample when requested."""
        patient_id, phase = self.samples[index]

        image_input, original_mask, pixel_spacing = self.frame_loader(
            patient_id,
            self.view,
            phase,
        )

        image = np.asarray(image_input.image)

        if image.ndim != 2:
            raise ValueError(
                f"{patient_id} {self.view} {phase}: "
                f"expected a 2D image, received {image.shape}"
            )

        if not np.issubdtype(image.dtype, np.number):
            raise ValueError(
                f"{patient_id} {self.view} {phase}: "
                "image must contain numeric values"
            )

        if not np.isfinite(image).all():
            raise ValueError(
                f"{patient_id} {self.view} {phase}: "
                "image contains NaN or infinite values"
            )

        if image.min() < 0 or image.max() > 255:
            raise ValueError(
                f"{patient_id} {self.view} {phase}: "
                "expected image intensities in [0, 255], got "
                f"[{float(image.min())}, {float(image.max())}]"
            )

        binary_mask = to_binary_lv_mask(original_mask)

        transformed_image, transformed_mask, geometry = (
            transform_image_and_mask(
                image,
                binary_mask,
                target_size=self.target_size,
            )
        )

        normalized_image = (
            transformed_image.astype(np.float32, copy=False) / 255.0
        )
        # "geometry" and "pixel_spacing" describe the
        # pre-augmentation 256x256 mapping; augmentation does not
        # update them.
        sample = {
            "image": normalized_image,
            "mask": transformed_mask,
            "patient_id": patient_id,
            "view": self.view,
            "phase": phase,
            "pixel_spacing": tuple(float(s) for s in pixel_spacing),
            "geometry": {
                name: getattr(geometry, name)
                for name in geometry.__dataclass_fields__
            },
        }

        if self.augment:
            augmented_image, augmented_mask, augmentation_params = (
                augment_image_and_mask(
                    normalized_image,
                    transformed_mask,
                    self._rng,
                    self._augment_config,
                )
            )
            sample["image"] = augmented_image
            sample["mask"] = augmented_mask
            sample["augmentation"] = augmentation_params

        return sample