import tempfile
import unittest
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from cv_pipeline.camus_dataset import CAMUSDataset
from cv_pipeline.contracts import ImageInput


def centre_offset(image, mask):
    """Largest axis distance (pixels) between the bright block and the mask."""
    bright = image > (125 / 255)
    if not bright.any() or not (mask == 1).any():
        raise AssertionError("bright block or mask is empty")
    bright_centre = np.argwhere(bright).mean(axis=0)
    mask_centre = np.argwhere(mask == 1).mean(axis=0)
    return float(np.abs(bright_centre - mask_centre).max())


class TestCAMUSDataset(unittest.TestCase):
    def create_split_files(self, root):
        contents = {
            "subgroup_training.txt": "patient0001\npatient0002\n",
            "subgroup_validation.txt": "patient0003\n",
            "subgroup_testing.txt": "patient0004\n",
        }

        for filename, content in contents.items():
            (root / filename).write_text(content, encoding="utf-8")

    def fake_frame_loader(self, patient_id, view, phase):
        image = np.full((12, 8), 128, dtype=np.float32)
        mask = np.zeros((12, 8), dtype=np.uint8)
        mask[3:9, 2:6] = 1

        # Mimic the values returned by load_camus_frame.
        image_input = ImageInput(
            image=image,
            patient_id=patient_id,
            modality="ECHO",
            view=view,
            frame_index=0,
        )

        return image_input, mask, (0.308, 0.308)

    def make_bright_region_loader(self, mask_columns):
        """Dim image (50) with a bright (200) block at columns 2:6.

        The mask is drawn at mask_columns. (2, 6) matches the bright block;
        any other value deliberately misaligns the mask.
        """
        start, stop = mask_columns

        def loader(patient_id, view, phase):
            image = np.full((12, 8), 50, dtype=np.float32)
            mask = np.zeros((12, 8), dtype=np.uint8)
            image[3:9, 2:6] = 200
            mask[3:9, start:stop] = 1

            image_input = ImageInput(
                image=image,
                patient_id=patient_id,
                modality="ECHO",
                view=view,
                frame_index=0,
            )
            return image_input, mask, (0.308, 0.308)

        return loader

    def make_constant_value_loader(self, value):
        def loader(patient_id, view, phase):
            image = np.full((12, 8), value, dtype=np.float32)
            mask = np.zeros((12, 8), dtype=np.uint8)
            mask[3:9, 2:6] = 1
            image_input = ImageInput(
                image=image,
                patient_id=patient_id,
                modality="ECHO",
                view=view,
                frame_index=0,
            )
            return image_input, mask, (0.308, 0.308)

        return loader

    def sample_from_loader(self, root, loader):
        dataset = CAMUSDataset(
            split="train",
            phases=("ED",),
            target_size=(16, 16),
            split_root=root,
            frame_loader=loader,
        )
        return dataset[0]

    def test_length_uses_patient_count_and_requested_phases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.create_split_files(root)

            dataset = CAMUSDataset(
                split="train",
                phases=("ED", "ES"),
                split_root=root,
                frame_loader=self.fake_frame_loader,
            )

            self.assertEqual(len(dataset), 4)

    def test_split_contains_only_its_official_patients(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.create_split_files(root)

            dataset = CAMUSDataset(
                split="validation",
                split_root=root,
                frame_loader=self.fake_frame_loader,
            )

            self.assertEqual(len(dataset), 2)
            self.assertEqual(
                {patient_id for patient_id, _ in dataset.samples},
                {"patient0003"},
            )

    def test_sample_has_normalized_image_binary_mask_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.create_split_files(root)

            dataset = CAMUSDataset(
                split="train",
                phases=("ED",),
                target_size=(16, 16),
                split_root=root,
                frame_loader=self.fake_frame_loader,
            )

            sample = dataset[0]

            self.assertEqual(sample["image"].shape, (16, 16))
            self.assertEqual(sample["mask"].shape, (16, 16))
            self.assertEqual(sample["image"].dtype, np.float32)
            self.assertEqual(sample["mask"].dtype, np.uint8)
            self.assertAlmostEqual(
                float(sample["image"].max()),
                128 / 255,
            )
            self.assertTrue(
                set(np.unique(sample["mask"])).issubset({0, 1})
            )
            self.assertEqual(sample["patient_id"], "patient0001")
            self.assertEqual(sample["view"], "4CH")
            self.assertEqual(sample["phase"], "ED")
            self.assertEqual(
                sample["pixel_spacing"],
                (0.308, 0.308),
            )

    def test_dataloader_batches_samples_and_geometry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.create_split_files(root)

            dataset = CAMUSDataset(
                split="train",
                phases=("ED", "ES"),
                target_size=(16, 16),
                split_root=root,
                frame_loader=self.fake_frame_loader,
            )

            batch = next(
                iter(
                    DataLoader(
                        dataset,
                        batch_size=2,
                        shuffle=False,
                    )
                )
            )

            self.assertEqual(
                tuple(batch["image"].shape),
                (2, 16, 16),
            )
            self.assertEqual(
                tuple(batch["mask"].shape),
                (2, 16, 16),
            )
            self.assertEqual(
                batch["patient_id"],
                ["patient0001", "patient0001"],
            )
            self.assertEqual(
                batch["phase"],
                ["ED", "ES"],
            )
            self.assertIn("original_height", batch["geometry"])
            self.assertEqual(
                tuple(batch["geometry"]["original_height"].shape),
                (2,),
            )

    def test_rejects_unknown_split(self):
        with self.assertRaisesRegex(ValueError, "Unknown split"):
            CAMUSDataset(split="development")

    def test_rejects_invalid_view(self):
        with self.assertRaisesRegex(ValueError, "view must be"):
            CAMUSDataset(view="3CH")

    def test_rejects_invalid_phase(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.create_split_files(root)

            with self.assertRaisesRegex(
                ValueError,
                "phases must contain",
            ):
                CAMUSDataset(
                    phases=("MID",),
                    split_root=root,
                    frame_loader=self.fake_frame_loader,
                )

    def test_mask_aligns_with_image_and_padding_is_zero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.create_split_files(root)

            sample = self.sample_from_loader(
                root, self.make_bright_region_loader((2, 6))
            )
            image, mask = sample["image"], sample["mask"]
            geometry = sample["geometry"]

            # The test only means something if padding actually exists.
            self.assertGreater(geometry["pad_left"], 0)
            self.assertGreater(geometry["pad_right"], 0)

            # Padding is zero in both image and mask.
            left = geometry["pad_left"]
            right = 16 - geometry["pad_right"]
            for array in (image, mask):
                self.assertTrue(np.all(array[:, :left] == 0))
                self.assertTrue(np.all(array[:, right:] == 0))

            # Mask is binary, not intensity-scaled, and not lost.
            self.assertEqual(int(mask.max()), 1)
            self.assertTrue(set(np.unique(mask)).issubset({0, 1}))

            # The bright image block and the mask coincide.
            self.assertLess(centre_offset(image, mask), 1.0)

    def test_alignment_check_detects_a_shifted_mask(self):
        """Proves the alignment measurement can fail.

        The mask is drawn 2 columns to the right of the bright block.
        After the 1.33x resize that is roughly 2.7 pixels.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.create_split_files(root)

            sample = self.sample_from_loader(
                root, self.make_bright_region_loader((4, 8))
            )

            offset = centre_offset(sample["image"], sample["mask"])
            self.assertGreater(offset, 2.0)

    def test_rejects_image_values_outside_0_255(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.create_split_files(root)

            for bad_value in (300, -1):
                with self.subTest(value=bad_value):
                    dataset = CAMUSDataset(
                        split="train",
                        phases=("ED",),
                        target_size=(16, 16),
                        split_root=root,
                        frame_loader=self.make_constant_value_loader(
                            bad_value
                        ),
                    )
                    with self.assertRaisesRegex(
                        ValueError, "expected image intensities"
                    ):
                        dataset[0]

    def test_accepts_boundary_values_0_and_255(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.create_split_files(root)

            sample = self.sample_from_loader(
                root, self.make_constant_value_loader(255)
            )
            self.assertAlmostEqual(float(sample["image"].max()), 1.0)

    def test_missing_file_error_names_patient_view_and_phase(self):
        """Uses the real adapter, so a missing file must fail loudly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.create_split_files(root)
            (root / "subgroup_training.txt").write_text(
                "patient9999\n", encoding="utf-8"
            )

            dataset = CAMUSDataset(
                split="train",
                view="4CH",
                phases=("ES",),
                split_root=root,
            )

            with self.assertRaises(FileNotFoundError) as context:
                dataset[0]

            message = str(context.exception)
            self.assertIn("patient9999", message)
            self.assertIn("4CH", message)
            self.assertIn("ES", message)


if __name__ == "__main__":
    unittest.main()