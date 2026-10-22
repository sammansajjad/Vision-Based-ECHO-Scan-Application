
import tempfile
import unittest
from pathlib import Path

from cv_pipeline.camus_splits import (
    get_split_patient_ids,
    load_split_patient_ids,
)


class TestCamusSplits(unittest.TestCase):
    def create_split_files(self, root):
        contents = {
            "subgroup_training.txt": "patient0001\npatient0002\n",
            "subgroup_validation.txt": "patient0003\n",
            "subgroup_testing.txt": "patient0004\n",
        }

        for filename, content in contents.items():
            (root / filename).write_text(content, encoding="utf-8")

    def test_loads_all_splits_with_expected_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.create_split_files(root)

            splits = load_split_patient_ids(root)

            self.assertEqual(
                set(splits),
                {"train", "validation", "test"},
            )
            self.assertEqual(
                splits["train"],
                ["patient0001", "patient0002"],
            )
            self.assertEqual(splits["validation"], ["patient0003"])
            self.assertEqual(splits["test"], ["patient0004"])

    def test_rejects_duplicate_ids_within_a_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.create_split_files(root)
            (root / "subgroup_training.txt").write_text(
                "patient0001\npatient0001\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_split_patient_ids(root)

    def test_rejects_overlap_between_splits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.create_split_files(root)
            (root / "subgroup_validation.txt").write_text(
                "patient0001\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "overlap"):
                load_split_patient_ids(root)

    def test_rejects_missing_split_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "subgroup_training.txt").write_text(
                "patient0001\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "Split file"):
                load_split_patient_ids(root)

    def test_returns_one_requested_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.create_split_files(root)

            patient_ids = get_split_patient_ids("validation", root)

            self.assertEqual(patient_ids, ["patient0003"])

    def test_rejects_unknown_split_name(self):
        with self.assertRaisesRegex(ValueError, "Unknown split"):
            get_split_patient_ids("development")


if __name__ == "__main__":
    unittest.main()
