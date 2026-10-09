import numpy as np


def laplacian_variance(image: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian over the image interior.

    Higher values mean more local intensity change (edges, texture, noise).
    The value depends on the image's intensity scale. The input is not modified.
    """
    array = np.asarray(image)

    if array.ndim != 2:
        raise ValueError(f"Expected a 2D image, received shape {array.shape}")
    if array.shape[0] < 3 or array.shape[1] < 3:
        raise ValueError("Image must be at least 3x3")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("Image must contain numeric pixel values")
    if not np.isfinite(array).all():
        raise ValueError("Image contains NaN or infinite pixel values")

    a = array.astype(np.float64)
    laplacian = (
        a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
        - 4.0 * a[1:-1, 1:-1]
    )
    return float(laplacian.var())