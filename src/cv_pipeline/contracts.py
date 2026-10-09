from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class ImageInput:
    """Generic representation of one image frame."""

    image: np.ndarray
    patient_id: str
    modality: str
    view: str
    frame_index: int


@dataclass
class ProcessedImage:
    """Preprocessed image plus a reference to its origin."""

    image: np.ndarray
    source: ImageInput


@dataclass
class QualityResult:
    """Decision about whether an image is usable."""

    is_usable: bool
    reason: str


@dataclass
class SegmentationMask:
    """Binary LV mask: 0 = background, 1 = LV target."""

    mask: np.ndarray
    source: ProcessedImage


@dataclass
class FeatureSet:
    """Named numerical features, each with a unit."""

    features: Dict[str, Tuple[float, str]]
    source: SegmentationMask


@dataclass
class CVResult:
    """Structured output of the CV subsystem."""

    status: str
    source: ImageInput
    quality_result: Optional[QualityResult] = None
    segmentation_mask: Optional[SegmentationMask] = None
    feature_set: Optional[FeatureSet] = None
    error_message: Optional[str] = None