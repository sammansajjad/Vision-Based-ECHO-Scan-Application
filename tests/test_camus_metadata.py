import tempfile
import unittest
from pathlib import Path

from cv_pipeline.camus_metadata import read_camus_image_quality, read_camus_info


class TestCamusMetadata(unittest.TestCase):
    def _make_root(self, text, view="4CH"):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patient_dir = Path(tmp.name) / "patient9999"
        patient_dir.mkdir()
        (patient_dir / f"Info_{view}.cfg").write_text(text, encoding="utf-8")
        return Path(tmp.name)

    def test_parses_keys_and_quality(self):
        root = self._make_root("ED: 1\nES: 21\nImageQuality: Good\n")
        info = read_camus_info("patient9999", root=root)
        self.assertEqual(info["ES"], "21")
        self.assertEqual(read_camus_image_quality("patient9999", root=root), "Good")

    def test_missing_file(self):
        root = self._make_root("ED: 1\n")
        with self.assertRaises(FileNotFoundError):
            read_camus_info("patient0000", root=root)

    def test_missing_quality(self):
        root = self._make_root("ED: 1\nES: 21\n")
        with self.assertRaises(ValueError):
            read_camus_image_quality("patient9999", root=root)

    def test_malformed_line(self):
        root = self._make_root("ED 1\n")
        with self.assertRaises(ValueError):
            read_camus_info("patient9999", root=root)

    def test_invalid_view(self):
        with self.assertRaises(ValueError):
            read_camus_info("patient0003", view="3CH")


if __name__ == "__main__":
    unittest.main()