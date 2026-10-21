
import unittest

import numpy as np

from cv_pipeline.model_input import (
    GeometryTransform,
    transform_image,
    transform_image_and_mask,
)


class TestTransformImage(unittest.TestCase):
    def test_preserves_aspect_ratio_and_records_geometry(self):
        image = np.ones((400, 300), dtype=np.float32)

        result, geometry = transform_image(image, (256, 256))

        self.assertEqual(result.shape, (256, 256))
        self.assertEqual(geometry.resized_height, 256)
        self.assertEqual(geometry.resized_width, 192)
        self.assertEqual(geometry.pad_left, 32)
        self.assertEqual(geometry.pad_right, 32)
        self.assertEqual(geometry.pad_top, 0)
        self.assertEqual(geometry.pad_bottom, 0)

    def test_rejects_invalid_image(self):
        with self.assertRaises(ValueError):
            transform_image(np.empty((0, 3)))

    def test_rejects_invalid_target_size(self):
        with self.assertRaises(ValueError):
            transform_image(np.ones((5, 5)), (0, 256))


class TestTransformImageAndMask(unittest.TestCase):
    def test_image_and_mask_share_geometry(self):
        image = np.arange(12 * 8, dtype=np.float32).reshape(12, 8)
        mask = np.zeros((12, 8), dtype=np.uint8)
        mask[3:9, 2:6] = 1
        original_mask = mask.copy()

        transformed_image, transformed_mask, geometry = (
            transform_image_and_mask(image, mask, (16, 16))
        )

        self.assertEqual(transformed_image.shape, (16, 16))
        self.assertEqual(transformed_mask.shape, (16, 16))
        self.assertIsInstance(geometry, GeometryTransform)
        np.testing.assert_array_equal(mask, original_mask)
        self.assertTrue(
            set(np.unique(transformed_mask)).issubset({0, 1})
        )

        # Padding is background and must be consistent with the geometry.
        if geometry.pad_top:
            self.assertTrue(
                np.all(transformed_mask[:geometry.pad_top, :] == 0)
            )
        if geometry.pad_bottom:
            self.assertTrue(
                np.all(transformed_mask[-geometry.pad_bottom:, :] == 0)
            )
        if geometry.pad_left:
            self.assertTrue(
                np.all(transformed_mask[:, :geometry.pad_left] == 0)
            )
        if geometry.pad_right:
            self.assertTrue(
                np.all(transformed_mask[:, -geometry.pad_right:] == 0)
            )

    def test_rejects_mismatched_shapes(self):
        image = np.ones((8, 8), dtype=np.float32)
        mask = np.zeros((7, 8), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "same spatial dimensions"):
            transform_image_and_mask(image, mask)

    def test_rejects_non_binary_mask(self):
        image = np.ones((8, 8), dtype=np.float32)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[0, 0] = 2

        with self.assertRaisesRegex(ValueError, "binary"):
            transform_image_and_mask(image, mask)


if __name__ == "__main__":
    unittest.main()
