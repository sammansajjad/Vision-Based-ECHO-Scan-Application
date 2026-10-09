import numpy as np

from cv_pipeline.contracts import ImageInput, ProcessedImage


def preprocess_image(source: ImageInput) -> ProcessedImage:
    """Validate and copy an image without changing its pixel values.

    This baseline intentionally performs no resizing, normalization,
    or contrast enhancement.
    """
    image = np.asarray(source.image)

    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, received shape {image.shape}")

    if image.size == 0:
        raise ValueError("Image must not be empty")

    if not np.issubdtype(image.dtype, np.number):
        raise ValueError("Image must contain numeric pixel values")

    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or infinite pixel values")

    return ProcessedImage(image=image.copy(), source=source)