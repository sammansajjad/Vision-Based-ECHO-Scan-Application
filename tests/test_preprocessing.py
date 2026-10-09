import unittest

import numpy as np

from cv_pipeline.contracts import ImageInput
from cv_pipeline.preprocessing import preprocess_image


class TestPreprocessImage(unittest.TestCase):
    def make_input(self, image):
        return ImageInput(
            image=image,
            patient_id="synthetic_patient",
            modality="ECHO",
            view="4CH",
            frame_index=0,
        )

    def test_preserves_image_values_and_metadata(self):
        image = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        source = self.make_input(image)

        result = preprocess_image(source)

        np.testing.assert_array_equal(result.image, image)
        self.assertEqual(result.image.shape, image.shape)
        self.assertEqual(result.image.dtype, image.dtype)
        self.assertIsNot(result.image, image)
        self.assertIs(result.source, source)

    def test_rejects_non_2d_image(self):
        source = self.make_input(np.zeros((2, 2, 3), dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "2D image"):
            preprocess_image(source)

    def test_rejects_empty_image(self):
        source = self.make_input(np.empty((0, 2), dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            preprocess_image(source)

    def test_rejects_non_numeric_image(self):
        source = self.make_input(np.array([["a", "b"]]))

        with self.assertRaisesRegex(ValueError, "numeric pixel values"):
            preprocess_image(source)

    def test_rejects_non_finite_pixels(self):
        source = self.make_input(np.array([[0.0, np.nan]], dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            preprocess_image(source)


if __name__ == "__main__":
    unittest.main()