from dataclasses import dataclass

import numpy as np
from typing import Dict, Tuple
from typing import Dict, Optional, Tuple


@dataclass
class ImageInput:
    """Generic representation of one image frame (contract Section 3)."""

    image: np.ndarray
    patient_id: str
    modality: str
    view: str
    frame_index: int


@dataclass
class ProcessedImage:
    """Preprocessed image plus a reference to its origin (contract Section 4)."""

    image: np.ndarray
    source: ImageInput


@dataclass
class QualityResult:
    """Decision about whether an image is usable (contract Section 5)."""

    is_usable: bool
    reason: str   


@dataclass
class SegmentationMask:
    """Binary LV mask: 0 = background, 1 = LV target (contract Section 6)."""

    mask: np.ndarray
    source: ProcessedImage     

@dataclass
class FeatureSet:
    """Named numerical features, each with a unit (contract Section 8)."""

    features: Dict[str, Tuple[float, str]]  # name -> (value, unit)
    source: SegmentationMask    


@dataclass
class CVResult:
    """Structured output of the CV subsystem (contract Section 9)."""

    status: str  # "ok", "unusable_image", or "processing_error"
    source: ImageInput
    quality_result: Optional[QualityResult] = None
    segmentation_mask: Optional[SegmentationMask] = None
    feature_set: Optional[FeatureSet] = None
    error_message: Optional[str] = None    