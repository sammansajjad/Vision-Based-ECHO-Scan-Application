\# Decision Record - Week 2



\## Decisions



1\. \*\*Array orientation.\*\* CAMUS arrays are kept exactly as loaded (patient0003: 591 x 487). For display only, they are transposed (`.T`), which puts the probe at the top. Evidence: visual check of patient0003 4CH ED; mask and image align. Only one patient was checked.

2\. \*\*Sharpness is not used for quality checks.\*\* EXP-003 found global Laplacian variance barely separates the expert labels (AUC 0.52 to 0.56).

3\. \*\*Quality checker is structural only.\*\* `check\_image\_quality` rejects only images with no intensity variation. `is\_usable=True` means "passed structural checks", not good quality.

4\. \*\*Pipeline statuses.\*\* `unusable\_image` (quality check rejected the image), `error` (software failure, recorded in `error\_message`), `quality\_passed` (placeholder until segmentation exists).



\## Still open



\- Exact LV target (cavity only vs cavity plus myocardium).

\- Whether ED/ES should be added to `ImageInput`.

\- Evidence-based criteria for image usability (EXP-004 idea: LV/myocardium contrast; uses ground truth, so exploratory only).

\- Whether invalid pixel data (NaN) should count as `unusable\_image` instead of `error`. It currently raises in preprocessing and is reported as `error`.

\- Pixel spacing is returned in array-axis order; after a transpose the order swaps. Equal for patient0003 (0.308, 0.308); other patients not checked.

\- In patient0003, the myocardium label extends outside the ultrasound sector; frequency across the dataset unknown.

