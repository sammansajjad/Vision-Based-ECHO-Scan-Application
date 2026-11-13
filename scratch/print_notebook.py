"""Throwaway read-only printer for CAMUS notebook text cells (CV-TASK-012)."""
import json

path = r"data\raw\CAMUS_public\jupyter\script_camus_ef.ipynb"
with open(path, encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    print(f"--- [{i}] [{cell['cell_type']}] ---")
    print("".join(cell["source"]))
    print()