
import unittest

import numpy as np

from cv_pipeline.mask_utils import to_binary_lv_mask


class TestToBinaryLVMask(unittest.TestCase):
    def test_converts_only_lv_cavity_to_foreground(self):
        original = np.array(
            [[0, 1, 2],
             [3, 1, 0]],
            dtype=np.uint8,
        )
        original_copy = original.copy()

        result = to_binary_lv_mask(original)

        expected = np.array(
            [[0, 1, 0],
             [0, 1, 0]],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, original.shape)
        self.assertTrue(set(np.unique(result)).issubset({0, 1}))
        np.testing.assert_array_equal(original, original_copy)

    def test_supports_masks_with_only_background(self):
        mask = np.zeros((3, 4), dtype=np.uint8)

        result = to_binary_lv_mask(mask)

        self.assertEqual(int(result.sum()), 0)

    def test_rejects_non_2d_mask(self):
        mask = np.zeros((2, 2, 1), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "2D mask"):
            to_binary_lv_mask(mask)

    def test_rejects_empty_mask(self):
        mask = np.empty((0, 2), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            to_binary_lv_mask(mask)

    def test_rejects_non_numeric_mask(self):
        mask = np.array([["0", "1"]])

        with self.assertRaisesRegex(ValueError, "numeric labels"):
            to_binary_lv_mask(mask)

    def test_rejects_non_finite_labels(self):
        mask = np.array([[0.0, np.nan]])

        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            to_binary_lv_mask(mask)

    def test_rejects_unexpected_labels(self):
        mask = np.array([[0, 1, 4]], dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "Unexpected CAMUS labels"):
            to_binary_lv_mask(mask)


if __name__ == "__main__":
    unittest.main()
