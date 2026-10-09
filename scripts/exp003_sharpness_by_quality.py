import csv
from pathlib import Path

import numpy as np

from cv_pipeline.camus_adapter import CAMUS_ROOT, load_camus_frame
from cv_pipeline.camus_metadata import read_camus_image_quality
from cv_pipeline.quality_metrics import laplacian_variance

VIEW, PHASE = "4CH", "ED"
OUT = Path("data/processed/exp003_sharpness_4ch_ed.csv")

rows, errors = [], []
patients = sorted(p.name for p in CAMUS_ROOT.glob("patient*") if p.is_dir())
for pid in patients:
    try:
        label = read_camus_image_quality(pid, VIEW)
        image_input, _, _ = load_camus_frame(pid, VIEW, PHASE)
        img = image_input.image.astype(np.float64)
        lo, hi = float(img.min()), float(img.max())
        scaled = (img - lo) / (hi - lo) if hi > lo else img * 0.0
        rows.append({
            "patient_id": pid, "quality": label,
            "min": lo, "max": hi,
            "lapvar_raw": laplacian_variance(img),
            "lapvar_minmax": laplacian_variance(scaled),
        })
    except (FileNotFoundError, ValueError) as exc:
        errors.append(f"{pid}: {exc}")

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"patients={len(patients)} ok={len(rows)} errors={len(errors)}")
for line in errors[:5]:
    print("  ", line)

mins = [r["min"] for r in rows]
maxs = [r["max"] for r in rows]
print(f"pixel min range: {min(mins):.1f}..{max(mins):.1f}; max range: {min(maxs):.1f}..{max(maxs):.1f}")


def auc(high, low):
    """P(random value from `high` group > random value from `low` group)."""
    h, l = np.asarray(high)[:, None], np.asarray(low)[None, :]
    return float(((h > l) + 0.5 * (h == l)).mean())


for key in ("lapvar_raw", "lapvar_minmax"):
    print(f"\n{key}")
    groups = {}
    for label in ("Good", "Medium", "Poor"):
        vals = np.array([r[key] for r in rows if r["quality"] == label])
        groups[label] = vals
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        print(f"  {label:6s} n={len(vals):3d} median={med:.4g} IQR=[{q1:.4g}, {q3:.4g}] "
              f"min={vals.min():.4g} max={vals.max():.4g}")
    print(f"  AUC Good>Poor   = {auc(groups['Good'], groups['Poor']):.3f}")
    print(f"  AUC Good>Medium = {auc(groups['Good'], groups['Medium']):.3f}")
    print(f"  AUC Medium>Poor = {auc(groups['Medium'], groups['Poor']):.3f}")