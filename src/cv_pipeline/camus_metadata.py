from pathlib import Path

from cv_pipeline.camus_adapter import CAMUS_ROOT


def read_camus_info(
    patient_id: str,
    view: str = "4CH",
    root: Path = CAMUS_ROOT,
) -> dict[str, str]:
    """Parse a CAMUS Info_<view>.cfg file into a {key: value} dict of strings."""
    view = view.upper()
    if view not in {"2CH", "4CH"}:
        raise ValueError("view must be '2CH' or '4CH'")

    path = Path(root) / patient_id / f"Info_{view}.cfg"
    if not path.is_file():
        raise FileNotFoundError(f"CAMUS info file not found: {path}")

    info: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"Malformed line in {path}: {line!r}")
        info[key.strip()] = value.strip()
    return info


def read_camus_image_quality(
    patient_id: str,
    view: str = "4CH",
    root: Path = CAMUS_ROOT,
) -> str:
    """Return the expert ImageQuality label exactly as written in the file."""
    info = read_camus_info(patient_id, view, root)
    if "ImageQuality" not in info:
        raise ValueError(f"ImageQuality missing for {patient_id} ({view})")
    return info["ImageQuality"]