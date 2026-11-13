import tempfile
import unittest
from pathlib import Path

import numpy as np

from cv_pipeline.augmentation import (
    AugmentationConfig,
    augment_image_and_mask,
)
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


def make_block_pair(size=64, block=(22, 42), value=0.8):
    """Zero background with a bright block; the mask covers the block."""
    image = np.zeros((size, size), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.uint8)
    image[block[0]:block[1], block[0]:block[1]] = value
    mask[block[0]:block[1], block[0]:block[1]] = 1
    return image, mask


class TestAugmentImageAndMask(unittest.TestCase):
    def test_image_and_mask_stay_synchronized(self):
        image, mask = make_block_pair()
        for seed in range(30):
            with self.subTest(seed=seed):
                out_image, out_mask, _ = augment_image_and_mask(
                    image, mask, np.random.default_rng(seed)
                )
                self.assertLess(centre_offset(out_image, out_mask), 1.0)

    def test_output_dtypes_and_value_ranges(self):
        image, mask = make_block_pair()
        image[0, 0] = 1.0  # exercise the upper clip with gain > 1
        for seed in range(30):
            with self.subTest(seed=seed):
                out_image, out_mask, _ = augment_image_and_mask(
                    image, mask, np.random.default_rng(seed)
                )
                self.assertEqual(out_image.dtype, np.float32)
                self.assertEqual(out_mask.dtype, np.uint8)
                self.assertEqual(out_image.shape, image.shape)
                self.assertEqual(out_mask.shape, mask.shape)
                self.assertGreaterEqual(float(out_image.min()), 0.0)
                self.assertLessEqual(float(out_image.max()), 1.0)
                self.assertTrue(set(np.unique(out_mask)).issubset({0, 1}))

    def test_zero_background_stays_exactly_zero(self):
        image, mask = make_block_pair()
        for seed in range(30):
            with self.subTest(seed=seed):
                out_image, _, _ = augment_image_and_mask(
                    image, mask, np.random.default_rng(seed)
                )
                # Far corners are untouched by the block after augmentation.
                self.assertTrue(np.all(out_image[:6, :6] == 0.0))
                self.assertTrue(np.all(out_image[-6:, -6:] == 0.0))

        # An all-zero image must stay all-zero (no additive offset).
        zeros = np.zeros((64, 64), dtype=np.float32)
        out_image, _, _ = augment_image_and_mask(
            zeros, mask, np.random.default_rng(0)
        )
        self.assertTrue(np.all(out_image == 0.0))

    def test_inputs_are_not_mutated_and_outputs_are_new(self):
        image, mask = make_block_pair()
        image_copy, mask_copy = image.copy(), mask.copy()
        out_image, out_mask, params = augment_image_and_mask(
            image, mask, np.random.default_rng(1)
        )
        np.testing.assert_array_equal(image, image_copy)
        np.testing.assert_array_equal(mask, mask_copy)
        self.assertFalse(np.shares_memory(out_image, image))
        self.assertFalse(np.shares_memory(out_mask, mask))
        self.assertTrue(params["applied"])

    def test_params_are_recorded_within_configured_ranges(self):
        image, mask = make_block_pair()
        config = AugmentationConfig()
        applied_count = 0
        for seed in range(40):
            _, _, params = augment_image_and_mask(
                image, mask, np.random.default_rng(seed), config
            )
            if not params["applied"]:
                continue
            applied_count += 1
            self.assertLessEqual(abs(params["angle_deg"]), 10.0)
            self.assertGreaterEqual(params["scale"], 0.9)
            self.assertLessEqual(params["scale"], 1.1)
            self.assertLessEqual(abs(params["translate_axis0"]), 0.04)
            self.assertLessEqual(abs(params["translate_axis1"]), 0.01)
            self.assertGreaterEqual(params["gain"], 0.9)
            self.assertLessEqual(params["gain"], 1.1)
            self.assertGreaterEqual(params["attempts"], 1)
            self.assertLessEqual(params["attempts"], 5)
        self.assertGreater(applied_count, 0)

    def test_guard_falls_back_to_unaugmented_copy(self):
        # A full-frame mask shrunk to half scale keeps only ~25% of its
        # foreground, so every attempt is rejected.
        config = AugmentationConfig(
            max_rotation_deg=0.0,
            scale_range=(0.5, 0.5),
            max_translation_axis0=0.0,
            max_translation_axis1=0.0,
            gain_range=(1.0, 1.0),
        )
        image = np.full((16, 16), 0.5, dtype=np.float32)
        mask = np.ones((16, 16), dtype=np.uint8)

        out_image, out_mask, params = augment_image_and_mask(
            image, mask, np.random.default_rng(0), config
        )

        self.assertFalse(params["applied"])
        self.assertEqual(params["attempts"], 5)
        np.testing.assert_array_equal(out_image, image)
        np.testing.assert_array_equal(out_mask, mask)
        self.assertFalse(np.shares_memory(out_image, image))
        self.assertFalse(np.shares_memory(out_mask, mask))

    def test_empty_mask_skips_the_guard(self):
        image, _ = make_block_pair()
        empty = np.zeros_like(image, dtype=np.uint8)
        _, out_mask, params = augment_image_and_mask(
            image, empty, np.random.default_rng(0)
        )
        self.assertTrue(params["applied"])
        self.assertEqual(params["attempts"], 1)
        self.assertEqual(int(out_mask.sum()), 0)

    def test_uses_only_the_passed_generator(self):
        image, mask = make_block_pair()
        np.random.seed(123)
        state_before = np.random.get_state()[1].copy()
        first = augment_image_and_mask(image, mask, np.random.default_rng(5))
        second = augment_image_and_mask(image, mask, np.random.default_rng(5))
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2], second[2])
        np.testing.assert_array_equal(
            np.random.get_state()[1], state_before
        )

    def test_rejects_invalid_inputs(self):
        image, mask = make_block_pair()
        rng = np.random.default_rng(0)

        bad_mask = mask.copy()
        bad_mask[0, 0] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            augment_image_and_mask(image, bad_mask, rng)

        for bad_value in (1.5, -0.1):
            bad_image = image.copy()
            bad_image[0, 0] = bad_value
            with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
                augment_image_and_mask(bad_image, mask, rng)

        nan_image = image.copy()
        nan_image[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            augment_image_and_mask(nan_image, mask, rng)

        with self.assertRaises(TypeError):
            augment_image_and_mask(image.astype(np.float64), mask, rng)
        with self.assertRaises(TypeError):
            augment_image_and_mask(image, mask.astype(np.int64), rng)
        with self.assertRaises(ValueError):
            augment_image_and_mask(image, mask[:, :-1], rng)
        with self.assertRaises(ValueError):
            augment_image_and_mask(image[None], mask[None], rng)
        with self.assertRaises(TypeError):
            augment_image_and_mask(image, mask, 123)
        with self.assertRaises(TypeError):
            augment_image_and_mask(image, mask, rng, config={})


class TestCAMUSDatasetAugmentation(unittest.TestCase):
    TARGET = (64, 64)

    def create_split_files(self, root):
        contents = {
            "subgroup_training.txt": "patient0001\npatient0002\n",
            "subgroup_validation.txt": "patient0003\n",
            "subgroup_testing.txt": "patient0004\n",
        }
        for filename, content in contents.items():
            (root / filename).write_text(content, encoding="utf-8")

    def block_loader(self, patient_id, view, phase):
        """Zero background with a bright (200) block and a matching mask."""
        image = np.zeros((48, 48), dtype=np.float32)
        mask = np.zeros((48, 48), dtype=np.uint8)
        image[14:34, 14:34] = 200
        mask[14:34, 14:34] = 1
        image_input = ImageInput(
            image=image,
            patient_id=patient_id,
            modality="ECHO",
            view=view,
            frame_index=0,
        )
        return image_input, mask, (0.308, 0.308)

    def make_dataset(self, root, **kwargs):
        return CAMUSDataset(
            split="train",
            phases=("ED", "ES"),
            target_size=self.TARGET,
            split_root=root,
            frame_loader=self.block_loader,
            **kwargs,
        )

    def all_samples(self, dataset):
        return [dataset[index] for index in range(len(dataset))]

    def test_augment_false_matches_default_and_adds_no_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.create_split_files(root)

            default = self.make_dataset(root)
            explicit = self.make_dataset(root, augment=False)

            for index in range(len(default)):
                a, b = default[index], explicit[index]
                self.assertEqual(list(a.keys()), list(b.keys()))
                self.assertNotIn("augmentation", a)
                self.assertNotIn("augmentation", b)
                np.testing.assert_array_equal(a["image"], b["image"])
                np.testing.assert_array_equal(a["mask"], b["mask"])
                for key in ("patient_id", "view", "phase",
                            "pixel_spacing", "geometry"):
                    self.assertEqual(a[key], b[key])

    def test_unaugmented_dataset_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.create_split_files(root)
            dataset = self.make_dataset(root, augment=False)

            first = dataset[0]
            second = dataset[0]
            np.testing.assert_array_equal(first["image"], second["image"])
            np.testing.assert_array_equal(first["mask"], second["mask"])

    def test_augmented_samples_are_synchronized_and_well_formed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.create_split_files(root)
            dataset = self.make_dataset(root, augment=True, augment_seed=3)

            baseline = self.make_dataset(root)[0]
            for _ in range(10):
                for sample in self.all_samples(dataset):
                    self.assertEqual(sample["image"].dtype, np.float32)
                    self.assertEqual(sample["mask"].dtype, np.uint8)
                    self.assertGreaterEqual(float(sample["image"].min()), 0)
                    self.assertLessEqual(float(sample["image"].max()), 1)
                    self.assertTrue(
                        set(np.unique(sample["mask"])).issubset({0, 1})
                    )
                    self.assertLess(
                        centre_offset(sample["image"], sample["mask"]), 1.0
                    )
                    # Geometry and spacing describe the pre-augmentation
                    # mapping and must be left untouched.
                    self.assertEqual(sample["geometry"], baseline["geometry"])
                    self.assertEqual(
                        sample["pixel_spacing"], baseline["pixel_spacing"]
                    )

    def test_augmentation_params_present_and_in_range(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.create_split_files(root)
            dataset = self.make_dataset(root, augment=True, augment_seed=0)

            applied = 0
            for _ in range(10):
                for sample in self.all_samples(dataset):
                    params = sample["augmentation"]
                    self.assertIn("applied", params)
                    if not params["applied"]:
                        continue
                    applied += 1
                    self.assertLessEqual(abs(params["angle_deg"]), 10.0)
                    self.assertTrue(0.9 <= params["scale"] <= 1.1)
                    self.assertLessEqual(abs(params["translate_axis0"]), 0.04)
                    self.assertLessEqual(abs(params["translate_axis1"]), 0.01)
                    self.assertTrue(0.9 <= params["gain"] <= 1.1)
            self.assertGreater(applied, 0)

    def test_same_seed_repeats_and_different_seed_differs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.create_split_files(root)

            run_a = self.all_samples(
                self.make_dataset(root, augment=True, augment_seed=7)
            )
            run_b = self.all_samples(
                self.make_dataset(root, augment=True, augment_seed=7)
            )
            run_c = self.all_samples(
                self.make_dataset(root, augment=True, augment_seed=8)
            )

            for a, b in zip(run_a, run_b):
                np.testing.assert_array_equal(a["image"], b["image"])
                np.testing.assert_array_equal(a["mask"], b["mask"])
                self.assertEqual(a["augmentation"], b["augmentation"])

            self.assertTrue(
                any(
                    not np.array_equal(a["image"], c["image"])
                    for a, c in zip(run_a, run_c)
                )
            )

    def test_rejects_invalid_augment_arguments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.create_split_files(root)

            for bad_augment in ("yes", 1, None):
                with self.subTest(augment=bad_augment):
                    with self.assertRaisesRegex(TypeError, "augment must"):
                        self.make_dataset(root, augment=bad_augment)

            for bad_seed in (True, 1.5, "3", None):
                with self.subTest(augment_seed=bad_seed):
                    with self.assertRaisesRegex(TypeError, "augment_seed"):
                        self.make_dataset(
                            root, augment=True, augment_seed=bad_seed
                        )

            with self.assertRaisesRegex(ValueError, "non-negative"):
                self.make_dataset(root, augment=True, augment_seed=-1)


if __name__ == "__main__":
    unittest.main()