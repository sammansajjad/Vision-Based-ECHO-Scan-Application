import unittest

import numpy as np

from cv_pipeline.contracts import ImageInput, ProcessedImage
from cv_pipeline.quality import check_image_quality


def make_processed(image):
    source = ImageInput(
        image=image, patient_id="patient9999",
        modality="ECHO", view="4CH", frame_index=0,
    )
    return ProcessedImage(image=image.copy(), source=source)


class TestCheckImageQuality(unittest.TestCase):
    def test_constant_image_is_unusable(self):
        result = check_image_quality(make_processed(np.full((10, 10), 7.0)))
        self.assertFalse(result.is_usable)
        self.assertTrue(result.reason)

    def test_all_zero_image_is_unusable(self):
        result = check_image_quality(make_processed(np.zeros((10, 10))))
        self.assertFalse(result.is_usable)

    def test_varied_image_passes_structural_checks(self):
        image = np.random.default_rng(0).random((10, 10))
        result = check_image_quality(make_processed(image))
        self.assertTrue(result.is_usable)
        self.assertIn("not assessed", result.reason)

    def test_does_not_modify_image(self):
        image = np.random.default_rng(1).random((10, 10)).astype(np.float32)
        processed = make_processed(image)
        before = processed.image.copy()
        check_image_quality(processed)
        self.assertTrue(np.array_equal(processed.image, before))

    def test_rejects_non_2d(self):
        with self.assertRaises(ValueError):
            check_image_quality(make_processed(np.zeros(10)))


if __name__ == "__main__":
    unittest.main()