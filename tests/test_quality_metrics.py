import unittest

import numpy as np

from cv_pipeline.quality_metrics import laplacian_variance


def box_blur(a):
    h, w = a.shape
    return sum(a[i:h - 2 + i, j:w - 2 + j] for i in range(3) for j in range(3)) / 9


class TestLaplacianVariance(unittest.TestCase):
    def test_constant_image_is_zero(self):
        self.assertEqual(laplacian_variance(np.full((10, 10), 5.0)), 0.0)

    def test_linear_gradient_is_zero(self):
        gradient = np.tile(np.arange(10, dtype=float), (10, 1))
        self.assertAlmostEqual(laplacian_variance(gradient), 0.0)

    def test_blur_lowers_value(self):
        rng = np.random.default_rng(0)
        sharp = rng.random((40, 40))
        self.assertLess(laplacian_variance(box_blur(sharp)), laplacian_variance(sharp))

    def test_does_not_modify_input(self):
        image = np.random.default_rng(1).random((8, 8)).astype(np.float32)
        before = image.copy()
        laplacian_variance(image)
        self.assertTrue(np.array_equal(image, before))

    def test_rejects_non_2d(self):
        with self.assertRaises(ValueError):
            laplacian_variance(np.zeros(10))

    def test_rejects_too_small(self):
        with self.assertRaises(ValueError):
            laplacian_variance(np.zeros((2, 5)))

    def test_rejects_non_finite(self):
        image = np.ones((5, 5))
        image[2, 2] = np.nan
        with self.assertRaises(ValueError):
            laplacian_variance(image)

    def test_rejects_non_numeric(self):
        with self.assertRaises(ValueError):
            laplacian_variance(np.array([["a", "b", "c"]] * 3))


if __name__ == "__main__":
    unittest.main()