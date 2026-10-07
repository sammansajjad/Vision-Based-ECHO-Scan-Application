import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# CAMUS 4CH image and ground-truth sequences
# --------------------------------------------------


patient_id = "patient0003"


base_path = (
    "data/raw/CAMUS_public/database_nifti/"
    f"{patient_id}/"
)
image_path = base_path + f"{patient_id}_4CH_half_sequence.nii.gz"
mask_path = base_path + f"{patient_id}_4CH_half_sequence_gt.nii.gz"

# Load data
images = nib.load(image_path).get_fdata()
masks = nib.load(mask_path).get_fdata()

print("Image shape:", images.shape)
print("Mask shape:", masks.shape)
print("Mask labels:", np.unique(masks))

# --------------------------------------------------
# ED = frame 1
# ES = frame 20
# Python uses zero-based indexing
# --------------------------------------------------

ed_image = images[:, :, 0]
ed_mask = masks[:, :, 0]

with open(base_path + "Info_4CH.cfg", "r") as file:
    for line in file:
        if line.startswith("ES:"):
            es_frame = int(line.split(":")[1].strip())
            break

es_image = images[:, :, es_frame - 1]
es_mask = masks[:, :, es_frame - 1]
print("Patient ID:", patient_id)
print("ED frame:", 1)
print("ES frame:", es_frame)

# --------------------------------------------------
# Visualize
# --------------------------------------------------

plt.figure(figsize=(12, 8))

# ED image
plt.subplot(2, 2, 1)
plt.imshow(ed_image, cmap="gray")
plt.title("ED - Original Image")
plt.axis("off")

# ED mask
plt.subplot(2, 2, 2)
plt.imshow(ed_mask)
plt.title("ED - Ground Truth")
plt.axis("off")

# ES image
plt.subplot(2, 2, 3)
plt.imshow(es_image, cmap="gray")
plt.title("ES - Original Image")
plt.axis("off")

# ES mask
plt.subplot(2, 2, 4)
plt.imshow(es_mask)
plt.title("ES - Ground Truth")
plt.axis("off")

plt.tight_layout()
plt.show()