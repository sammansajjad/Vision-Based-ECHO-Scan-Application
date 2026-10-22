import unittest

import numpy as np

from cv_pipeline.metrics import dice_coefficient, iou_score


def example_masks():
    truth = np.zeros((4, 4), dtype=np.uint8)
    truth[1:3, 1:3] = 1          # 4 pixels

    prediction = np.zeros((4, 4), dtype=np.uint8)
    prediction[1, 1:4] = 1       # 3 pixels, 2 overlap with truth
    return prediction, truth


class TestHandCheckedExample(unittest.TestCase):
    def test_dice_matches_hand_calculation(self):
        prediction, truth = example_masks()
        self.assertAlmostEqual(dice_coefficient(prediction, truth), 4 / 7)

    def test_iou_matches_hand_calculation(self):
        prediction, truth = example_masks()
        self.assertAlmostEqual(iou_score(prediction, truth), 2 / 5)

    def test_iou_and_dice_satisfy_their_relation(self):
        prediction, truth = example_masks()
        dice = dice_coefficient(prediction, truth)
        self.assertAlmostEqual(iou_score(prediction, truth), dice / (2 - dice))


class TestKnownCases(unittest.TestCase):
    def test_identical_masks_score_one(self):
        _, truth = example_masks()
        self.assertEqual(dice_coefficient(truth, truth), 1.0)
        self.assertEqual(iou_score(truth, truth), 1.0)

    def test_disjoint_masks_score_zero(self):
        _, truth = example_masks()
        prediction = np.zeros((4, 4), dtype=np.uint8)
        prediction[0, 0] = 1
        self.assertEqual(dice_coefficient(prediction, truth), 0.0)
        self.assertEqual(iou_score(prediction, truth), 0.0)

    def test_empty_prediction_scores_zero(self):
        _, truth = example_masks()
        prediction = np.zeros((4, 4), dtype=np.uint8)
        self.assertEqual(dice_coefficient(prediction, truth), 0.0)
        self.assertEqual(iou_score(prediction, truth), 0.0)

    def test_boolean_masks_are_accepted(self):
        prediction, truth = example_masks()
        self.assertAlmostEqual(
            dice_coefficient(prediction.astype(bool), truth.astype(bool)),
            4 / 7,
        )

    def test_inputs_are_not_modified(self):
        prediction, truth = example_masks()
        prediction_before, truth_before = prediction.copy(), truth.copy()
        dice_coefficient(prediction, truth)
        iou_score(prediction, truth)
        np.testing.assert_array_equal(prediction, prediction_before)
        np.testing.assert_array_equal(truth, truth_before)


class TestInvalidInput(unittest.TestCase):
    def test_empty_ground_truth_raises(self):
        prediction, _ = example_masks()
        empty = np.zeros((4, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            dice_coefficient(prediction, empty)
        with self.assertRaises(ValueError):
            iou_score(prediction, empty)

    def test_both_masks_empty_raises(self):
        empty = np.zeros((4, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            dice_coefficient(empty, empty)
        with self.assertRaises(ValueError):
            iou_score(empty, empty)

    def test_rejects_non_array(self):
        _, truth = example_masks()
        with self.assertRaises(TypeError):
            dice_coefficient(truth.tolist(), truth)

    def test_rejects_non_2d(self):
        _, truth = example_masks()
        with self.assertRaises(ValueError):
            dice_coefficient(truth.reshape(-1), truth)

    def test_rejects_zero_size_array(self):
        _, truth = example_masks()
        with self.assertRaises(ValueError):
            dice_coefficient(np.zeros((0, 4), dtype=np.uint8), truth)

    def test_rejects_non_numeric(self):
        _, truth = example_masks()
        with self.assertRaises(TypeError):
            dice_coefficient(np.full((4, 4), "a"), truth)

    def test_rejects_non_finite(self):
        prediction, truth = example_masks()
        bad = prediction.astype(float)
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            dice_coefficient(bad, truth)

    def test_rejects_non_binary_labels(self):
        prediction, truth = example_masks()
        multi_class = prediction.copy()
        multi_class[0, 0] = 2
        with self.assertRaises(ValueError):
            dice_coefficient(multi_class, truth)

    def test_rejects_shape_mismatch(self):
        prediction, truth = example_masks()
        with self.assertRaises(ValueError):
            dice_coefficient(prediction[:, :3], truth)


if __name__ == "__main__":
    unittest.main()