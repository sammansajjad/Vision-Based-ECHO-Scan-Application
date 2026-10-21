# Experiment Log

## EXP-001 - CAMUS 4CH Sequence and Ground-Truth Inspection

### Purpose

Understand the relationship between ECHO 4CH image sequences and their ground-truth segmentation masks.

### Dataset

CAMUS public dataset.

### Patients Inspected

- `patient0001`
- `patient0002`
- `patient0003`

### Observed Evidence

- The inspected 4CH image data contains multiple cardiac frames.
- Spatial dimensions varied between the three inspected patients:
  - `patient0001`: 549 × 389
  - `patient0002`: 748 × 584
  - `patient0003`: 591 × 487
- Sequence lengths varied between the three inspected patients:
  - `patient0001`: 20 frames
  - `patient0002`: 15 frames
  - `patient0003`: 21 frames
- For all three inspected patients, image and ground-truth sequence dimensions matched.
- Ground-truth masks contained labels 0, 1, 2, and 3:
  - Label 0: background
  - Label 1: LV cavity
  - Label 2: LV myocardium
  - Label 3: left atrium
- ED and ES correspond to different cardiac frames.
- The ES frame number is patient-specific and is available from `Info_4CH.cfg`.
- The inspection script was updated to read the ES frame from metadata instead of assuming a fixed frame number.
- For `patient0001`, the LV cavity pixel count was:
  - ED: 33,363 pixels
  - ES: 18,261 pixels

### Interpretation

The inspected CAMUS examples show that both spatial dimensions and sequence length can vary between patients. Therefore, the inspection pipeline should not assume a fixed image size or a fixed number of frames.

The ES frame should be obtained from patient metadata rather than assumed to occur at a fixed frame index.

The LV cavity pixel counts at ED and ES differed for `patient0001`, with fewer LV cavity pixels at ES. These values are 2D pixel counts and are not physical volume measurements.

### Limitations

- Only the first three inspected patients are included in this experiment.
- These observations must not be generalized to the entire CAMUS dataset without further inspection.

### Status

Completed.

---

## EXP-002 - CAMUS Expert Image-Quality Label Survey

### Purpose

Identify which expert image-quality labels exist in CAMUS and how common they are before designing an automatic quality check.

### Method

`scripts/survey_camus_quality.py` reads `ImageQuality` from `Info_2CH.cfg` and `Info_4CH.cfg` for every patient folder, using `read_camus_image_quality`. No images were read and no raw data was modified.

### Observed Evidence

- 500 patient folders were found; there were 0 read errors.
- 4CH:
  - Good: 288 (57.6%)
  - Medium: 165 (33.0%)
  - Poor: 47 (9.4%)
- 2CH:
  - Good: 217 (43.4%)
  - Medium: 214 (42.8%)
  - Poor: 69 (13.8%)

### Interpretation

CAMUS provides a three-level expert quality label per patient and view. Poor images are a minority in 4CH, so any later comparison must report group sizes and must not rely on overall accuracy alone.

### Limitations

- The label is expert-assigned, and its exact criteria are not documented in the project files inspected.
- It is not yet known whether image measurements, such as sharpness or contrast, separate Good from Poor.
- This survey does not establish clinical validity.

### Status

Completed.

---

## EXP-003 - Global Sharpness (Laplacian Variance) vs Expert Quality Label

### Purpose

Test whether a simple whole-image sharpness measure separates CAMUS Good, Medium, and Poor images.

### Method

`scripts/exp003_sharpness_by_quality.py` computed `laplacian_variance` from `src/cv_pipeline/quality_metrics.py` on the unmodified ED frame of every 4CH patient, using two variants: raw pixel values and per-image min-max scaling. Scaling was performed inside the experiment only.

Groups were compared by median, interquartile range (IQR), and area under the receiver operating characteristic curve (AUC). Per-patient values were saved to `data/processed/exp003_sharpness_4ch_ed.csv`, which is git-ignored.

### Observed Evidence

- 500 of 500 patients processed, with 0 errors.
- Pixel minimum was 0 for every frame; pixel maximum ranged from 204 to 255.
- Raw variant median:
  - Good: 315.6 (n = 288)
  - Medium: 301.0 (n = 165)
  - Poor: 295.7 (n = 47)
- Raw AUC:
  - Good vs Poor: 0.552
  - Good vs Medium: 0.532
  - Medium vs Poor: 0.522
- Min-max variant AUC:
  - Good vs Poor: 0.555
  - Good vs Medium: 0.525
  - Medium vs Poor: 0.535
- Group ranges overlapped substantially.

### Interpretation

Global Laplacian variance was only slightly better than chance at separating the expert quality groups in this experiment. The results did not justify a threshold, and no sharpness threshold was added to the pipeline.

### Limitations

- Only 4CH ED frames and one sharpness metric were tested.
- The measure includes the area outside the ultrasound sector and speckle noise.
- Confidence intervals were not computed; the Poor group was small (n = 47).
- The criteria behind the expert quality labels are not known.

### Status

Completed. Negative result.

---

## EXP-004 - Full CAMUS Image–Mask Integrity Audit

### Purpose

Verify the structural integrity of the dedicated CAMUS end-diastolic (ED) and end-systolic (ES) image and ground-truth mask pairs across both 2CH and 4CH views for all patients in the official dataset split.

### Method

`scripts/audit_camus_dataset.py` loaded the official training, validation, and testing patient lists and checked the corresponding patient directories.

For each patient, the script checked the dedicated image–mask pairs for both views and both phases. Checks included file existence, NIfTI loading, 2D dimensionality, matching image and mask shapes, matching affine matrices, finite array values, and mask labels restricted to `{0, 1, 2, 3}`. The audit did not modify raw dataset files.

### Observed Evidence

- Training split: 400 patients.
- Validation split: 50 patients.
- Testing split: 50 patients.
- Unique patient IDs across all splits: 500.
- Duplicate IDs within split files: 0.
- Overlap between split files: 0.
- Patient directories missing from the split lists: 0.
- Split IDs missing corresponding patient directories: 0.
- Expected dedicated image–mask pairs checked: 2,000.
- Pairs with errors: 0.
- Reported errors: 0.
- Audit result: PASS.

### Interpretation

All 2,000 expected dedicated image–mask pairs passed the structural checks implemented in the audit script. The official patient split lists were disjoint and collectively corresponded to all 500 patient directories found on disk.

The official patient-level split provides a structurally verified starting partition for subsequent experiments. This audit alone does not establish annotation suitability or the validity of any evaluation protocol.

### Limitations

- The audit verifies structural integrity, including file availability, successful NIfTI loading, dimensional consistency, image–mask shape and affine agreement, finite array values, and permitted annotation labels. It does not establish anatomical correctness, annotation quality, or clinical suitability.
- Binary LV-cavity mask conversion was implemented in `src/cv_pipeline/mask_utils.py` and covered by unit tests. CAMUS label `1` is mapped to foreground; all other valid CAMUS labels are mapped to background.
- Image and binary-mask transformation was implemented in `src/cv_pipeline/model_input.py` using aspect-ratio-preserving resizing, centered zero-padding, linear interpolation for images, and nearest-neighbor interpolation for masks.
- Paired image–mask transformation was integration-tested on six selected CAMUS 4CH frames: ED and ES for `patient0001`, `patient0002`, and `patient0003`. The transformation was not audited across all 2,000 image–mask pairs.
- The reported test suite covers unit tests and selected real-data integration tests. It does not measure segmentation-model performance.
- No segmentation-accuracy, diagnostic-performance, or clinical-validity claim is supported by these experiments.

### Status

Completed: structural dataset audit and initial preprocessing/segmentation-input preparation.

The full test suite passed 42 tests, including unit tests and real-data integration tests. This verifies the behavior covered by those tests; it does not establish segmentation accuracy or clinical validity.