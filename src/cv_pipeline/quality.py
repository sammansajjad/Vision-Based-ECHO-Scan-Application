import numpy as np

from cv_pipeline.contracts import ProcessedImage, QualityResult


def check_image_quality(processed: ProcessedImage) -> QualityResult:
    """Structural usability check. Never modifies the image.

    Only detects an image with no intensity variation (blank or constant).
    Passing does NOT imply good clinical quality: EXP-003 found that global
    sharpness does not separate the expert quality labels, so no
    sharpness or contrast threshold is used.
    """
    image = np.asarray(processed.image)

    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, received shape {image.shape}")

    if float(image.max()) == float(image.min()):
        return QualityResult(
            is_usable=False,
            reason="Image has no intensity variation (constant or blank)",
        )

    return QualityResult(
        is_usable=True,
        reason="Passed structural checks only; image quality not assessed",
    )