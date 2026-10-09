import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file, no window needed
import matplotlib.pyplot as plt
import numpy as np

from cv_pipeline.camus_adapter import load_camus_frame

patient_id = sys.argv[1] if len(sys.argv) > 1 else "patient0003"
phase = sys.argv[2] if len(sys.argv) > 2 else "ED"

image_input, mask, spacing = load_camus_frame(patient_id, "4CH", phase)
image = image_input.image
image_view, mask_view = image.T, mask.T  # display only; loaded data unchanged

out = Path("data/processed/figures") / f"{patient_id}_4CH_{phase}.png"
out.parent.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
axes[0].imshow(image_view, cmap="gray")
axes[0].set_title("Image (transposed for display)")
axes[1].imshow(mask_view, cmap="viridis", vmin=0, vmax=3, interpolation="nearest")
axes[1].set_title("Mask labels 0-3 (transposed for display)")
axes[2].imshow(image_view, cmap="gray")
overlay = np.ma.masked_where(mask_view == 0, mask_view)
axes[2].imshow(overlay, cmap="viridis", vmin=0, vmax=3, alpha=0.4, interpolation="nearest")
axes[2].set_title("Overlay")
for ax in axes:
    ax.axis("off")
fig.suptitle(
    f"{patient_id} 4CH {phase} | loaded shape={image.shape} | spacing={spacing}"
)
fig.tight_layout()
fig.savefig(out, dpi=100)
print(f"Saved: {out}")