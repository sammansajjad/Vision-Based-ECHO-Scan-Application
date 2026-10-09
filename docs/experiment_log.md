\# Experiment Log



\## EXP-001 - CAMUS 4CH Sequence and Ground-Truth Inspection



\### Purpose



Understand the relationship between ECHO 4CH image sequences and their ground-truth segmentation masks.



\### Dataset



CAMUS public dataset.



\### Patients Inspected



\- patient0001

\- patient0002

\- patient0003



\### Observed Evidence



\- The inspected 4CH image data contains multiple cardiac frames.

\- Spatial dimensions varied between the three inspected patients:

&#x20; - patient0001: 549 x 389

&#x20; - patient0002: 748 x 584

&#x20; - patient0003: 591 x 487

\- Sequence lengths varied between the three inspected patients:

&#x20; - patient0001: 20 frames

&#x20; - patient0002: 15 frames

&#x20; - patient0003: 21 frames

\- For all three inspected patients, image and ground-truth sequence dimensions matched.

\- Ground-truth masks contained labels 0, 1, 2, and 3.

\- Label 0 represents background.

\- Label 1 represents LV cavity.

\- Label 2 represents LV myocardium.

\- Label 3 represents left atrium.

\- ED and ES correspond to different cardiac frames.

\- The ES frame number is patient-specific and is available from `Info\_4CH.cfg`.

\- The inspection script was updated to read the ES frame from metadata instead of assuming a fixed frame number.

\- For patient0001, the LV cavity pixel count was:

&#x20; - ED: 33,363 pixels

&#x20; - ES: 18,261 pixels



\### Interpretation



The inspected CAMUS examples show that both spatial dimensions and sequence length can vary between patients. Therefore, the inspection pipeline should not assume a fixed image size or fixed number of frames.



The ES frame should be obtained from patient metadata rather than assumed to occur at a fixed frame index.



The LV cavity pixel counts at ED and ES differed for patient0001, with fewer LV cavity pixels at ES. These values are 2D pixel counts and are not physical volume measurements.



\### Limitations



Only the first three inspected patients are included in this experiment.



These observations must not be generalized to the entire CAMUS dataset without further inspection.



\### Status



Completed.

## EXP-002 - CAMUS Expert Image-Quality Label Survey

### Purpose

Find out which expert image-quality labels exist in CAMUS and how common they are, before designing any automatic quality check.

### Method

`scripts/survey_camus_quality.py` reads `ImageQuality` from `Info_2CH.cfg` and `Info_4CH.cfg` for every patient folder, using `read_camus_image_quality`. No images were read and no raw data was modified.

### Observed Evidence

- 500 patient folders were found; there were 0 read errors.
- 4CH: Good 288 (57.6%), Medium 165 (33.0%), Poor 47 (9.4%).
- 2CH: Good 217 (43.4%), Medium 214 (42.8%), Poor 69 (13.8%).

### Interpretation

CAMUS provides a three-level expert quality label per patient and view. Poor images are a small minority in 4CH, so any later comparison must report group sizes and must not rely on overall accuracy alone.

### Limitations

- The label is expert-assigned and its exact criteria are not documented in our files.
- It is not yet known whether any image measurement (e.g. sharpness or contrast) separates Good from Poor.
- This survey says nothing about clinical validity.

### Status

Completed.

## EXP-003 - Global Sharpness (Laplacian Variance) vs Expert Quality Label

### Purpose

Test whether a simple whole-image sharpness measure separates CAMUS Good, Medium and Poor images.

### Method

`scripts/exp003_sharpness_by_quality.py` computed `laplacian_variance` (`src/cv_pipeline/quality_metrics.py`) on the unmodified ED frame of every 4CH patient, in two variants: raw pixel values and per-image min-max scaling (scaling done inside the experiment only). Groups were compared by median, IQR and AUC. Per-patient values were saved to `data/processed/exp003_sharpness_4ch_ed.csv` (git-ignored).

### Observed Evidence

- 500 of 500 patients processed, 0 errors. Pixel minimum was 0 everywhere; pixel maximum ranged 204 to 255.
- Raw variant, median: Good 315.6 (n=288), Medium 301.0 (n=165), Poor 295.7 (n=47).
- Raw AUC: Good>Poor 0.552, Good>Medium 0.532, Medium>Poor 0.522.
- Min-max variant AUC: Good>Poor 0.555, Good>Medium 0.525, Medium>Poor 0.535.
- Group ranges overlap almost completely.

### Interpretation

Global Laplacian variance is only slightly better than chance at separating the expert quality groups. No threshold is justified, and none was added to the pipeline.

### Limitations

- Only 4CH ED frames and one metric were tested.
- The measure includes the area outside the ultrasound sector and speckle noise.
- No confidence intervals were computed; the Poor group is small (n=47).
- The criteria behind the expert label are not known.

### Status

Completed. Negative result.

