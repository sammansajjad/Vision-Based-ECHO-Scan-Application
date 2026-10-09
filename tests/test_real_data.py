import unittest

import numpy as np

from cv_pipeline.camus_adapter import CAMUS_ROOT, load_camus_frame
from cv_pipeline.pipeline import STATUS_QUALITY_PASSED, run_pipeline

PATIENTS = ["patient0001", "patient0002", "patient0003"]


@unittest.skipUnless(CAMUS_ROOT.is_dir(), "CAMUS data not available")
class TestRealCamusFrames(unittest.TestCase):
    def test_pipeline_on_real_frames(self):
        for patient_id in PATIENTS:
            for phase in ("ED", "ES"):
                with self.subTest(patient=patient_id, phase=phase):
                    image_input, mask, spacing = load_camus_frame(patient_id, "4CH", phase)
                    self.assertEqual(image_input.image.shape, mask.shape)
                    self.assertTrue(set(np.unique(mask)).issubset({0, 1, 2, 3}))
                    self.assertTrue(all(s > 0 for s in spacing))

                    result = run_pipeline(image_input)
                    self.assertEqual(result.status, STATUS_QUALITY_PASSED)
                    self.assertIs(result.source, image_input)
                    self.assertIsNone(result.error_message)


if __name__ == "__main__":
    unittest.main()