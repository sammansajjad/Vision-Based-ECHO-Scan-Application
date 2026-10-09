import unittest

import numpy as np

from cv_pipeline.contracts import ImageInput
from cv_pipeline.pipeline import (
    STATUS_ERROR,
    STATUS_QUALITY_PASSED,
    STATUS_UNUSABLE,
    run_pipeline,
)


def make_input(image):
    return ImageInput(
        image=image, patient_id="patient9999",
        modality="ECHO", view="4CH", frame_index=0,
    )


class TestRunPipeline(unittest.TestCase):
    def test_constant_image_is_unusable(self):
        source = make_input(np.full((10, 10), 3.0))
        result = run_pipeline(source)
        self.assertEqual(result.status, STATUS_UNUSABLE)
        self.assertFalse(result.quality_result.is_usable)
        self.assertIsNone(result.segmentation_mask)
        self.assertIsNone(result.feature_set)
        self.assertIsNone(result.error_message)

    def test_varied_image_passes_quality(self):
        source = make_input(np.random.default_rng(0).random((10, 10)))
        result = run_pipeline(source)
        self.assertEqual(result.status, STATUS_QUALITY_PASSED)
        self.assertTrue(result.quality_result.is_usable)
        self.assertIsNone(result.error_message)

    def test_software_failure_is_reported_as_error(self):
        image = np.ones((10, 10))
        image[0, 0] = np.nan
        result = run_pipeline(make_input(image))
        self.assertEqual(result.status, STATUS_ERROR)
        self.assertTrue(result.error_message)
        self.assertIsNone(result.quality_result)

    def test_source_identity_is_preserved(self):
        source = make_input(np.random.default_rng(1).random((10, 10)))
        result = run_pipeline(source)
        self.assertIs(result.source, source)


if __name__ == "__main__":
    unittest.main()