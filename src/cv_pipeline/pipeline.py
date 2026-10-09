from cv_pipeline.contracts import CVResult, ImageInput
from cv_pipeline.preprocessing import preprocess_image
from cv_pipeline.quality import check_image_quality

STATUS_UNUSABLE = "unusable_image"
STATUS_ERROR = "error"
STATUS_QUALITY_PASSED = "quality_passed"  # placeholder: segmentation not implemented yet


def run_pipeline(source: ImageInput) -> CVResult:
    """Preprocess, then quality check. Segmentation/features are not built yet."""
    try:
        processed = preprocess_image(source)
        quality = check_image_quality(processed)
    except Exception as exc:  # boundary: report software failures, do not crash
        return CVResult(
            status=STATUS_ERROR,
            source=source,
            error_message=f"{type(exc).__name__}: {exc}",
        )

    if not quality.is_usable:
        return CVResult(status=STATUS_UNUSABLE, source=source, quality_result=quality)

    return CVResult(status=STATUS_QUALITY_PASSED, source=source, quality_result=quality)