import unittest

import torch

from cv_pipeline.prediction import logits_to_binary_masks


class TestLogitsToBinaryMasks(unittest.TestCase):
    def test_converts_logits_to_binary_masks(self):
        logits = torch.tensor([[[[-2.0, 0.0, 2.0]]]])

        result = logits_to_binary_masks(logits)

        expected = torch.tensor([[[[0, 1, 1]]]], dtype=torch.uint8)
        self.assertTrue(torch.equal(result, expected))

    def test_output_is_uint8(self):
        logits = torch.randn(1, 1, 4, 4)

        result = logits_to_binary_masks(logits)

        self.assertEqual(result.dtype, torch.uint8)

    def test_output_contains_only_zero_and_one(self):
        logits = torch.randn(2, 1, 8, 8)

        result = logits_to_binary_masks(logits)

        self.assertTrue(torch.all((result == 0) | (result == 1)).item())

    def test_rejects_non_tensor_input(self):
        with self.assertRaises(TypeError):
            logits_to_binary_masks([[0.0, 1.0]])

    def test_rejects_wrong_dimensions(self):
        with self.assertRaises(ValueError):
            logits_to_binary_masks(torch.randn(4, 4))

    def test_rejects_non_finite_values(self):
        logits = torch.tensor([[[[float("nan")]]]])

        with self.assertRaises(ValueError):
            logits_to_binary_masks(logits)

    def test_rejects_threshold_out_of_range(self):
        logits = torch.zeros(1, 1, 2, 2)

        with self.assertRaises(ValueError):
            logits_to_binary_masks(logits, threshold=1.5)


if __name__ == "__main__":
    unittest.main()