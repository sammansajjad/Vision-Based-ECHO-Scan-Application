
import unittest

import numpy as np

from cv_pipeline.camus_adapter import CAMUS_ROOT, load_camus_frame
from cv_pipeline.mask_utils import to_binary_lv_mask
from cv_pipeline.model_input import transform_image_and_mask
from cv_pipeline.pipeline import STATUS_QUALITY_PASSED, run_pipeline


PATIENTS = ["patient0001", "patient0002", "patient0003"]


@unittest.skipUnless(CAMUS_ROOT.is_dir(), "CAMUS data not available")
class TestRealCamusFrames(unittest.TestCase):
    def test_pipeline_on_real_frames(self):
        for patient_id in PATIENTS:
            for phase in ("ED", "ES"):
                with self.subTest(patient=patient_id, phase=phase):
                    image_input, mask, spacing = load_camus_frame(
                        patient_id, "4CH", phase
                    )

                    self.assertEqual(image_input.image.shape, mask.shape)
                    self.assertTrue(
                        set(np.unique(mask)).issubset({0, 1, 2, 3})
                    )
                    self.assertTrue(all(s > 0 for s in spacing))

                    result = run_pipeline(image_input)

                    self.assertEqual(
                        result.status, STATUS_QUALITY_PASSED
                    )
                    self.assertIs(result.source, image_input)
                    self.assertIsNone(result.error_message)

    def test_binary_mask_and_geometry_on_real_frames(self):
        for patient_id in PATIENTS:
            for phase in ("ED", "ES"):
                with self.subTest(patient=patient_id, phase=phase):
                    image_input, original_mask, spacing = load_camus_frame(
                        patient_id, "4CH", phase
                    )

                    # Convert the original CAMUS annotation into an
                    # LV-cavity-only binary mask.
                    binary_mask = to_binary_lv_mask(original_mask)

                    # Resize and pad the image and mask using the same
                    # geometry, with interpolation appropriate to each.
                    transformed_image, transformed_mask, geometry = (
                        transform_image_and_mask(
                            image_input.image,
                            binary_mask,
                            target_size=(256, 256),
                        )
                    )

                    # Verify transformed dimensions.
                    self.assertEqual(
                        transformed_image.shape, (256, 256)
                    )
                    self.assertEqual(
                        transformed_mask.shape, (256, 256)
                    )
                    self.assertEqual(
                        transformed_image.shape,
                        transformed_mask.shape,
                    )

                    # Verify binary-mask properties.
                    self.assertEqual(transformed_mask.dtype, np.uint8)
                    self.assertTrue(
                        set(np.unique(transformed_mask)).issubset({0, 1})
                    )

                    # Verify the recorded geometry describes the
                    # original image dimensions.
                    self.assertEqual(
                        geometry.original_height,
                        image_input.image.shape[0],
                    )
                    self.assertEqual(
                        geometry.original_width,
                        image_input.image.shape[1],
                    )

                    # Confirm valid physical spacing.
                    self.assertTrue(all(s > 0 for s in spacing))

                    # Confirm the adapter's original annotation remains
                    # multi-class and has not been overwritten.
                    self.assertTrue(
                        set(np.unique(original_mask)).issubset(
                            {0, 1, 2, 3}
                        )
                    )


if __name__ == "__main__":
    unittest.main()
