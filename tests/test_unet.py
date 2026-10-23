import unittest

import torch

from cv_pipeline.models.unet import UNet


class TestUNetForwardPass(unittest.TestCase):
    def test_output_shape_and_dtype(self):
        model = UNet()
        model.eval()

        images = torch.randn(2, 1, 256, 256)

        with torch.no_grad():
            logits = model(images)

        self.assertEqual(tuple(logits.shape), (2, 1, 256, 256))
        self.assertEqual(logits.dtype, torch.float32)

    def test_output_values_are_finite(self):
        model = UNet()
        model.eval()

        images = torch.randn(1, 1, 256, 256)

        with torch.no_grad():
            logits = model(images)

        self.assertTrue(torch.isfinite(logits).all().item())


if __name__ == "__main__":
    unittest.main()