
from pathlib import Path


DEFAULT_SPLIT_ROOT = Path("data/raw/CAMUS_public/database_split")

SPLIT_FILES = {
    "train": "subgroup_training.txt",
    "validation": "subgroup_validation.txt",
    "test": "subgroup_testing.txt",
}


def load_split_patient_ids(
    root: Path = DEFAULT_SPLIT_ROOT,
) -> dict[str, list[str]]:
    """Load and validate all official CAMUS patient splits."""
    root = Path(root)
    split_ids = {}

    for split_name, filename in SPLIT_FILES.items():
        path = root / filename

        if not path.is_file():
            raise FileNotFoundError(f"Split file not found: {path}")

        patient_ids = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if len(patient_ids) != len(set(patient_ids)):
            raise ValueError(f"Duplicate patient IDs in {path}")

        split_ids[split_name] = patient_ids

    all_ids = [
        patient_id
        for patient_ids in split_ids.values()
        for patient_id in patient_ids
    ]

    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Patient IDs overlap across official split files")

    return split_ids


def get_split_patient_ids(
    split: str,
    root: Path = DEFAULT_SPLIT_ROOT,
) -> list[str]:
    """Return patient IDs for one official split."""
    if split not in SPLIT_FILES:
        raise ValueError(
            f"Unknown split {split!r}; expected one of "
            f"{tuple(SPLIT_FILES)}"
        )

    return load_split_patient_ids(root)[split]
