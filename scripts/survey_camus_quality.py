from collections import Counter

from cv_pipeline.camus_adapter import CAMUS_ROOT
from cv_pipeline.camus_metadata import read_camus_image_quality

for view in ("2CH", "4CH"):
    counts = Counter()
    errors = []
    patients = sorted(p.name for p in CAMUS_ROOT.glob("patient*") if p.is_dir())
    for patient_id in patients:
        try:
            counts[read_camus_image_quality(patient_id, view)] += 1
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{patient_id}: {exc}")
    print(f"{view}: patients={len(patients)} labels={dict(counts)} errors={len(errors)}")
    for line in errors[:5]:
        print("  ", line)