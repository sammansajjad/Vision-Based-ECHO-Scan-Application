# Week 2 Progress Report



Project: Computer Vision Based Assistive Application for Analyzing ECHO Scans



## 1. Objective



Build the computer-vision foundation before any model training: data loading, preprocessing, image-quality checking, visualization and a small pipeline orchestrator.



## 2. Work Completed



- CAMUS adapter (src/cv_pipeline/camus_adapter.py): loads one 4CH/2CH frame at ED or ES with its ground-truth mask and pixel spacing. It validates the view, phase, file existence, matching shapes and matching geometry. Original mask labels are preserved and images are not resized or normalized.

- Cleaned the shared contracts (duplicate imports removed, fields unchanged).

- Baseline preprocessing: validates the input and returns an unchanged copy. No resizing, normalization or enhancement.

- CAMUS metadata reader (src/cv_pipeline/camus_metadata.py): reads Info_<view>.cfg, including the expert ImageQuality label.

- EXP-002: surveyed the expert quality labels of all 500 CAMUS patients.

- EXP-003: tested whether global sharpness (Laplacian variance) separates the expert quality groups.

- Structural quality checker (src/cv_pipeline/quality.py) and a pipeline orchestrator (src/cv_pipeline/pipeline.py).

- Visualization script (scripts/visualize_camus_frame.py).

- Decision record (docs/decision_record_week2.md).

- 28 automated tests passed in one run (27 synthetic-data tests and 1 real-data test covering 3 patients, ED and ES).

- Work is committed and pushed to the project repository.



## 3. Findings



- All 500 patients have a quality label in both views. 4CH: Good 288, Medium 165, Poor 47. 2CH: Good 217, Medium 214, Poor 69. No read errors.

- Global sharpness barely separates the labels. In the 4CH ED frames, the AUC values were between 0.52 and 0.56 (0.5 would be chance). The group ranges overlap almost completely. No sharpness threshold was therefore added.

- The CAMUS arrays are stored transposed relative to the usual display. After transposing for display only, the patient0003 image looks like a standard four-chamber view and the mask aligns with it.

- On the three inspected patients (6 frames), the pipeline ran end to end and returned the status quality_passed.



## 4. Limitations



- The quality checker is structural only. It rejects an image with no intensity variation, and passing it does not mean good clinical quality.

- EXP-003 covers only 4CH ED frames and one metric. No confidence intervals were computed, and the Poor group is small (n=47).

- The orientation and mask-alignment check is a visual check of one patient.

- The real-data test covers only three patients.

- Segmentation and feature extraction are not implemented. No model has been trained and no accuracy results exist.

- Preprocessing has no options yet. This is a deliberate choice, since no experiment has justified any enhancement.

- Pixel counts are not physical measurements. No clinical validity is claimed.



## 5. Open Questions



- Exact LV target: cavity only or cavity plus myocardium.

- Whether ED/ES should be added to ImageInput.

- Evidence-based criteria for image usability.

- Whether invalid pixel data should count as unusable_image instead of error.

- Pixel-spacing axis order after transposing.

- In patient0003, the myocardium label extends outside the ultrasound sector; how often this happens is unknown.



## 6. Next Steps



- Decide the LV target and verify the CAMUS label mapping.

- Create a patient-level train/validation/test split.

- Establish a baseline segmentation experiment before any optimization.

