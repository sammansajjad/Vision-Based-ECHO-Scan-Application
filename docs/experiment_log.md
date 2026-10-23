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

---

## CAMUS Orientation and Model-Input Geometry Decision

### Purpose

Investigate whether the transpose used in CAMUS visualization indicates an orientation problem in the loaded image arrays, and determine whether changes are required before implementing the LV segmentation training loader.

### Method

The following components were inspected:

- `src/cv_pipeline/camus_adapter.py`
- `scripts/visualize_camus_frame.py`
- `src/cv_pipeline/model_input.py`
- `tests/test_model_input.py`
- `tests/test_real_data.py`

For `patient0003_4CH_ED.nii.gz`, the NIfTI array shape, affine matrix, axis codes, and header spacing were inspected. The existing visualization and model-input transformation tests were also reviewed.

### Observed Evidence

- The loaded image array shape was `(591, 487)`.
- The reported header zooms were approximately `(0.308, 0.308)` mm along the two array axes.
- `nib.aff2axcodes()` returned `('L', 'P', 'S')` for the image affine. The image array itself is two-dimensional; the third code describes the third coordinate direction represented by the affine, not a third array dimension.
- The adapter loads the image and mask without an explicit transpose or flip.
- The adapter checks that the image and mask have matching shapes and affine matrices.
- `visualize_camus_frame.py` transposes both arrays for display using `image.T` and `mask.T`. The loaded arrays remain unchanged.
- Visual inspection confirmed that the displayed LV mask aligned with the visible LV cavity for the inspected frame.
- All six tests in `tests.test_model_input` passed.
- The geometry transformation preserves aspect ratio, records resized dimensions and padding, applies matching geometry to the image and binary mask, uses linear interpolation for images and nearest-neighbor interpolation for masks, and keeps padding as background in the mask.

### Interpretation

The available evidence does not demonstrate an orientation error in the loaded CAMUS arrays. The transpose currently used by the visualization script is a display operation applied consistently to both the image and mask.

For the initial LV segmentation baseline, the loaded arrays will therefore be preserved without an additional orientation correction. The existing aspect-ratio-preserving resize-and-padding transformation will be reused for model input.

The initial evaluation will compare predictions and ground-truth LV masks in the same transformed coordinate system. Physical-coordinate mapping is outside the scope of this initial preprocessing step.

### Limitations

- Visual mask alignment does not independently establish anatomical orientation or validate the complete mapping between voxel coordinates and physical coordinates.
- The inspection used a selected frame and did not constitute a dataset-wide orientation audit.
- The existing geometry tests do not implement or verify an inverse transformation from model-input coordinates to the original image grid.
- The transformation does not use the NIfTI affine or physical spacing to map predictions into physical coordinates.
- Physical measurements and overlays on the original image grid will require a separately specified and tested coordinate-mapping procedure.

### Decision

**No orientation-related code change is made at this stage.**

The current array-loading and display conventions are retained. Inverse geometry mapping and physical-coordinate interpretation are deferred until required for prediction visualization or measurement extraction.

This decision is limited to the initial preprocessing and segmentation baseline. It does not claim that anatomical orientation or physical-coordinate mapping has been fully validated.

---

## EXP-005 - CAMUS Dataset Loader and PyTorch Batching

### Purpose

Implement and verify a reusable dataset loader for binary left-ventricular (LV) cavity segmentation using the official CAMUS patient splits. Prepare image-mask pairs for a future segmentation model while preserving image geometry and patient-level separation.

### Dataset

CAMUS public dataset.

- Training: 400 patients.
- Validation: 50 patients.
- Testing: 50 patients.
- Views supported by the loader: 2CH and 4CH. The audit below covers 4CH only.
- Cardiac phases: ED and ES (dedicated `_ED` and `_ES` files; see EXP-007).
- Default model-input size: 256 × 256.

### Method

The following components were implemented or extended:

- `src/cv_pipeline/camus_splits.py` loads and validates the official patient split files, and maps `train`, `validation` and `test` to the three official files.
- `src/cv_pipeline/camus_dataset.py` implements an on-demand PyTorch-compatible dataset. It reuses `load_camus_frame`, `to_binary_lv_mask` and `transform_image_and_mask`.
- `src/cv_pipeline/mask_utils.py` converts multiclass CAMUS annotations into binary LV-cavity masks.
- `src/cv_pipeline/model_input.py` resizes images and masks while preserving aspect ratio and recording padding geometry.
- `tests/test_camus_splits.py` verifies split loading, duplicate and overlap rejection, a missing split file, and unknown split names.
- `tests/test_camus_dataset.py` verifies dataset behavior using synthetic data and an injected frame loader.
- `scripts/audit_camus_dataset.py` uses the shared patient-split loader.
- `scripts/audit_camus_loader.py` audits the loader one split at a time on real data.

Samples are `(patient, phase)` pairs built from each split's patient list, so both phases of a patient always stay in that patient's split. Each sample is loaded on demand. The loader raises an error if an image has non-finite values or values outside [0, 255]; it never clips. Images are then divided by 255.0 and returned as float32 arrays of shape (H, W). Binary masks are uint8 with background 0 and LV cavity 1. No extra transpose is applied.

Image resizing uses linear interpolation and mask resizing uses nearest-neighbor interpolation. Centered zero-padding brings both arrays to the target dimensions. Geometry metadata and pixel spacing are returned with each sample.

`scripts/audit_camus_loader.py` loads every 4CH ED and ES sample of a split, records every failure instead of stopping at the first, and checks each loaded sample directly: output shapes, finite image values, mask values, non-empty LV mask, zero padding in image and mask, and the presence of both phases per patient. It reads data only. The test split was audited for integrity only, under the rule that no preprocessing or design choice may be changed because of what test frames look like.

### Observed Evidence

Loader audit, 4CH, ED and ES:

| Split | Patients | Samples enumerated | Loaded | Failed | Contract problems |
|---|---:|---:|---:|---:|---:|
| Training | 400 | 800 | 800 | 0 | 0 |
| Validation | 50 | 100 | 100 | 0 | 0 |
| Testing | 50 | 100 | 100 | 0 | 0 |

- Patients with both ED and ES loaded: 400 of 400, 50 of 50, and 50 of 50.
- Mask values across all samples in every split: exactly `{0, 1}`. No empty LV mask.
- Source image sizes (height × width ranges): training 323–1181 × 292–973; validation 387–865 × 325–714; testing 354–865 × 292–714.
- LV pixels per mask after resizing to 256 × 256: training min 910, median 4,608, max 16,578; validation min 1,631, median 4,998, max 13,593; testing min 1,757, median 4,331, max 13,127.
- Scaled image range in every split: [0.0, 1.0]. The [0, 255] guard never fired.
- An earlier inline check of all 800 training frames found float32 integer-valued images with global minimum 0, global maximum 255, and per-frame maxima from 196 to 255.
- Earlier structural audit (EXP-004): 2,000 dedicated image-mask pairs with 0 reported errors.
- A two-sample DataLoader batch of synthetic samples produced image and mask tensors of shape 2 × 16 × 16, with patient identifiers, phase labels and geometry metadata collated.
- Full automated test suite: 86 tests passed.

### Interpretation

The loader provides a reusable interface between CAMUS data and a future segmentation model. Official patient-level splits stay separate, and nothing is loaded into memory up front. All 1,000 expected 4CH ED/ES samples load end to end and satisfy the output contract that the audit measures.

The measured intensity range supports division by 255.0 for the CAMUS 4CH frames checked. The shared geometric transformation applies the same spatial mapping to each image and its mask.

Synthetic tests include a check that a deliberately shifted mask is detected by the alignment measurement, and a check that padding is zero in image and mask.

### Limitations

- No segmentation model result comes from this experiment.
- Non-empty masks, valid shapes and ranges do not establish annotation correctness or clinical suitability.
- Orientation was checked visually on selected frames only (see the orientation section). No inverse mapping to the original image grid exists, and physical measurements are not validated.
- Source frames shrink by up to about 4×, and `INTER_LINEAR` does not antialias. This is a possible preprocessing question for a later experiment.
- The dataset enforces nothing to keep the test split from being used. The rule is a convention.
- Missing or malformed files surface when a sample is loaded. The audit script is the explicit pre-training check, and it should be re-run if the data changes.
- 2CH was not audited.
- The current PyTorch environment is CPU-only.

### Status

Completed: dataset-loader implementation, real-data audit of all 4CH ED/ES samples in the three official splits, and automated regression testing.

The full test suite passed 86 tests. This verifies the behavior covered by those tests; it does not establish segmentation accuracy or clinical validity.

---

## EXP-006 - U-Net Training Smoke Test and Checkpoint Verification

### Purpose

Verify that the initial U-Net training script can perform a limited training run, evaluate the model on validation data, save a checkpoint, and load the checkpoint successfully.

### Dataset

CAMUS public dataset, 4CH view, ED and ES phases.

- Training samples available: 800.
- Validation samples available: 100.
- Input dimensions: 256 × 256.
- Device: CPU.
- Model: 2D U-Net with 1,942,289 parameters.

### Method

`scripts/train_unet.py` was implemented to train the U-Net using the Adam optimizer and binary cross-entropy with logits loss (`BCEWithLogitsLoss`).

The smoke test used:

- Epochs: 1.
- Batch size: 2.
- Maximum training batches: 2.
- Maximum validation batches: 2.
- Learning rate: 0.001.
- Random seed: 42.

These are the script's default arguments, and the checkpoint metadata confirms epoch 1 and seed 42. The batch limits are not stored in the checkpoint.

After validation, the script saved the model and optimizer state, epoch number, validation metrics, and seed to `outputs/checkpoints/unet_4ch_latest.pth`.

The saved checkpoint was then loaded using PyTorch with CPU mapping and `weights_only=True`.

### Observed Evidence

- Training script syntax check passed.
- The smoke test completed without a reported error.
- Training loss: 0.7371.
- Validation loss: 0.7310.
- Validation Dice: 0.1262.
- Validation IoU: 0.0678.
- Checkpoint file was created successfully; reported file size was 23,412,339 bytes.
- Checkpoint loading succeeded.
- The loaded checkpoint contained `epoch`, `model_state_dict`, `optimizer_state_dict`, `seed`, and `val_metrics`.
- The saved epoch was 1.
- Loaded validation metrics matched the training output.
- Reading `scripts/train_unet.py` shows it builds only the `train` and `validation` loaders and does not use the test split.
- A single-frame visualization (`scripts/evaluate_one_frame.py`, validation split, patient0200, 4CH, ED) gave Dice 0.1257 and IoU 0.0671, and its predicted mask was uniform across the whole frame.
- A follow-up check on the four validation frames scored by the smoke test (patient0200 and patient0206, ED and ES) found that the checkpoint predicts foreground for 100% of pixels in every frame, while the ground-truth LV fraction was 4.3% to 10.2%. The checkpoint is therefore an all-foreground predictor, and its Dice and IoU reflect the LV's share of the frame, not segmentation ability.
- A second limited smoke test used 1 epoch, batch size 4, a maximum of 25 training batches and 5 validation batches. It reported training loss 0.5860, validation loss 0.6923, Dice 0.2411 and IoU 0.1397. These validation metrics cover only 20 frames processed in file order and are not a full validation evaluation.
- On the four inspected validation frames, this checkpoint predicted foreground for approximately 47.5%–50.2% of pixels, compared with a ground-truth LV fraction of 4.3%–10.2%. This remains evidence of poor foreground estimation, not useful segmentation performance.
- The second smoke test overwrote `outputs/checkpoints/unet_4ch_latest.pth`. A backup of the first smoke-test checkpoint has been retained by the researcher; its filename and integrity have not been independently verified here.

### Interpretation
The Dice and IoU values come from two highly restricted smoke tests. In the first run, only two training batches were processed, resulting in at most two optimizer updates. The second run processed up to 25 training batches, but still covered only a small portion of the training data. Neither run constitutes full training. The reported Dice and IoU values must not be treated as reliable segmentation performance or compared with published results. Neither checkpoint is established as a valid segmentation baseline.

### Limitations
- The first smoke test processed two training batches and two validation batches, covering four validation frames in file order without shuffling.
- The second smoke test processed up to 25 training batches and 5 validation batches, covering at most 20 validation frames in file order without shuffling. These results are not a full validation evaluation.
- The model has not undergone full training.
- No final test-set evaluation was performed.
- No comparison with a baseline or published method was performed.
- The script hard-codes the loss, optimizer, learning rate and seed, and overwrites the same checkpoint file every epoch, with no best-checkpoint selection. These are smoke-test settings, not decided protocol (see EXP-008).
- `set_seed` seeds Python, NumPy and PyTorch only, with no determinism flags.
- No claims about clinical usefulness or diagnostic performance are supported.
- Resuming training from the saved optimizer state has not yet been tested.
- Checkpoints are stored under `outputs/`, which is covered by `.gitignore` and should not be committed to the repository.

### Status

Completed: initial training smoke test and checkpoint save/load verification.

Full training, model-performance evaluation, prediction visualization, and final test-set assessment remain pending.

---

## EXP-007 - Dedicated ED Masks vs Half-Sequence Frame 0 Masks

### Purpose

Explain why the LV cavity pixel count in EXP-001 (patient0001 ED: 33,363) differs from the count obtained through the CAMUS adapter (33,394), and decide which annotation source the baseline will use.

### Method

Read-only comparison of frame index 0 of `<patient>_4CH_half_sequence_gt.nii.gz` with the dedicated `<patient>_4CH_ED_gt.nii.gz` masks, and of the corresponding images for patient0001. Dice was computed on the binary LV cavity (label 1) using `cv_pipeline.metrics.dice_coefficient`. No raw data was modified.

### Observed Evidence

- The EXP-001 count (33,363) was reproduced from the half-sequence file. The dedicated ED mask gives 33,394.
- For patient0001, the two images are pixel-identical; 654 mask pixels differ across all labels.
- patient0001 label counts (background, LV cavity, myocardium, left atrium):
  - half-sequence: 152,626 / 33,363 / 16,156 / 11,416
  - dedicated ED: 152,593 / 33,394 / 16,164 / 11,410
- LV cavity Dice, half-sequence frame 0 vs dedicated ED:
  - patient0001: 0.9959 (654 differing pixels)
  - patient0002: 0.9988 (280 differing pixels)
  - patient0003: 0.9951 (628 differing pixels)

### Interpretation

The two annotation sources are not identical but agree closely on the LV cavity in the three patients compared. The comparison establishes agreement only. It does not show that either source is more accurate.

### Decision

The dedicated `_ED` and `_ES` image and ground-truth files are the chosen source for baseline training and evaluation. The half-sequence masks are excluded from baseline training and evaluation, so that annotation sources are not mixed. This is a baseline design choice, not a finding about annotation quality.

### Limitations

- Only three patients at ED were compared. Agreement has not been established across all patients or for ES.
- The location and cause of the differing pixels were not examined.
- Images were compared for patient0001 only.
- Which source is more accurate was not assessed.

### Status

Completed as a three-patient comparison. It is not a validation of annotation consistency across CAMUS.

---

## EXP-008 - LV Cavity Segmentation Baseline Protocol (Planned)

### Purpose

Fix the baseline design choices before the first real training run, so later changes can be compared against it. These are baseline design choices, not experimentally proven optimal settings.

### Data and Splits

- CAMUS 4CH, dedicated `_ED` and `_ES` image and `_gt` mask files (see EXP-007).
- The official patient-level splits (400 training, 50 validation, 50 testing patients) are applied to patient IDs before any frames are loaded. Both the ED and ES frames of a patient stay in that patient's split.
- Measured frame counts (EXP-005 audit): 800 training, 100 validation and 100 test frames.
- Training uses ED and ES frames jointly. This is a decided baseline choice.
- Target: LV cavity (label 1) via `to_binary_lv_mask`; all other labels are background.

### Input Preparation (baseline choices)

- Target size 256 × 256 via `transform_image_and_mask`: aspect-ratio-preserving resize with centered zero-padding. Images use linear interpolation and masks use nearest-neighbor interpolation.
- Intensity: `image / 255.0`, applied to images only. Masks are never scaled. No normalization statistics are fitted, so no validation or test data influences preprocessing.
- A later comparison with per-image z-score normalization is a separate experiment, using the same patient splits and evaluation metrics.
- Intensity-range prerequisite: completed. The range was 0 to 255 on all 800 training frames (inline check) and on all 1,000 4CH ED/ES frames in the loader audit (EXP-005).

### Evaluation

- Per-image Dice and IoU (`cv_pipeline.metrics`), reported as mean and spread, separately for ED and ES. Empty ground truth raises an error; an empty prediction scores 0.
- Evaluation is patient-separated. Model selection and tuning use the validation split only.
- The 50 test patients (100 frames) remain untouched until the model configuration is frozen, and are then used once for final evaluation.
- Integrity audit rule for the test split: problems are handled by fixing data handling or documenting the exception. No normalization, target size or other design choice may change based on what test frames look like.

### Not Yet Decided

- Model architecture (a U-Net exists in the repository; its use as the baseline is not yet recorded as a decision).
- Loss function, optimizer, learning rate, batch size, number of epochs, random seed and determinism settings. The smoke test in EXP-006 used Adam, BCE with logits and learning rate 0.001 as hard-coded values, but those are not recorded as protocol choices.
- Epoch-selection and best-checkpoint strategy.
- Augmentation.
- Spacing handling and physical-unit measurements.

These will be recorded before the first real training run.

### Claims Not Made

No segmentation accuracy, diagnostic performance or clinical validity is claimed.

### Status

Planned. No full training or evaluation run has been performed. The only trained checkpoint is the one-epoch smoke test in EXP-006.