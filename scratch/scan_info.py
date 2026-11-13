"""Throwaway read-only scan of CAMUS Info cfg files (CV-TASK-012).

Reads only. Writes nothing. Reuses read_camus_info from the project.
"""
import statistics
from collections import Counter
from pathlib import Path

from cv_pipeline.camus_adapter import CAMUS_ROOT
from cv_pipeline.camus_metadata import read_camus_info

SPLIT_DIR = Path(r"data\raw\CAMUS_public\database_split")


def to_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


splits = {}
for name in ("training", "validation", "testing"):
    lines = (SPLIT_DIR / f"subgroup_{name}.txt").read_text().splitlines()
    splits[name] = [l.strip() for l in lines if l.strip()]

all_ids = [p for ids in splits.values() for p in ids]
split_of = {p: n for n, ids in splits.items() for p in ids}
print("Patients listed in split files:", len(all_ids), "unique:", len(set(all_ids)))

info4 = {}
info2 = {}
errors = []
for pid in all_ids:
    try:
        info4[pid] = read_camus_info(pid, "4CH", CAMUS_ROOT)
    except Exception as exc:
        errors.append((pid, "4CH", repr(exc)))
    try:
        info2[pid] = read_camus_info(pid, "2CH", CAMUS_ROOT)
    except Exception as exc:
        errors.append((pid, "2CH", repr(exc)))

print("\n== Read results ==")
print("4CH parsed:", len(info4), "of", len(all_ids))
print("2CH parsed:", len(info2), "of", len(all_ids))
print("Read errors:", len(errors))
for e in errors[:10]:
    print("  ", e)

print("\n== Key sets (4CH) ==")
for keys, n in Counter(tuple(sorted(i.keys())) for i in info4.values()).items():
    print(n, keys)

print("\n== ImageQuality (4CH) ==")
q_all = Counter(i.get("ImageQuality", "<MISSING>") for i in info4.values())
print("overall:", dict(q_all), "denominator:", len(info4))
for name in splits:
    c = Counter(
        info4[p].get("ImageQuality", "<MISSING>") for p in splits[name] if p in info4
    )
    print(name + ":", dict(c))

print("\n== EF (4CH) ==")
ef_raw = {p: i.get("EF") for p, i in info4.items()}
missing = [p for p, v in ef_raw.items() if v is None or v == ""]
ef = {p: to_float(v) for p, v in ef_raw.items() if v not in (None, "")}
non_numeric = [p for p, v in ef.items() if v is None]
ef = {p: v for p, v in ef.items() if v is not None}
print("missing:", len(missing), "non-numeric:", len(non_numeric), "numeric:", len(ef))
if ef:
    vals = sorted(ef.values())
    print("min", vals[0], "max", vals[-1])
    print("mean", round(statistics.mean(vals), 3), "median", statistics.median(vals))
    print("unique values:", len(set(vals)))
    print("all whole numbers:", all(v == int(v) for v in vals))
    print("descriptive 10-wide bins (not clinical categories):")
    bins = Counter(int(v // 10) * 10 for v in vals)
    for b in sorted(bins):
        print("  ", b, "-", b + 9.99, ":", bins[b])
    for name in splits:
        sv = [ef[p] for p in splits[name] if p in ef]
        if sv:
            print(name, "n", len(sv), "mean", round(statistics.mean(sv), 2),
                  "min", min(sv), "max", max(sv))

print("\n== ED/ES/NbFrame sanity (4CH) ==")
bad = 0
for p, i in info4.items():
    ed, es, nb = to_float(i.get("ED")), to_float(i.get("ES")), to_float(i.get("NbFrame"))
    if ed is None or es is None or nb is None or not (1 <= ed <= nb and 1 <= es <= nb):
        bad += 1
print("patients with missing/non-numeric/out-of-range ED/ES/NbFrame:", bad)

print("\n== 4CH vs 2CH agreement per patient ==")
both = [p for p in info4 if p in info2]
for key in ("EF", "Age", "Sex", "ImageQuality", "ED", "ES", "NbFrame"):
    same = sum(1 for p in both if info4[p].get(key) == info2[p].get(key))
    print(key, "identical in", same, "of", len(both))