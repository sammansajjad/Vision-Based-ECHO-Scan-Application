# EXP-011 — Qualitative Investigation of Difficult LV Segmentation Cases



**Project:** Computer Vision Based Assistive Application for Analyzing ECHO Scans  

**Experiment ID:** EXP-011  

**Status:** Completed — validation-only failure-case analysis  

**Task:** 4CH left-ventricular (LV) cavity segmentation  

**Dataset:** CAMUS public dataset  

**Evaluation split:** Official CAMUS validation split  

**Model:** Existing U-Net baseline  

**Checkpoint:** `outputs/checkpoints/unet_4ch_best_full_validation.pth`  

**Prediction threshold:** 0.50  



---



## 1. Objective



EXP-011 investigated selected difficult LV cavity segmentation cases identified through the patient-level performance analysis in EXP-010.



EXP-010 showed that segmentation performance varied across the 50 patients in the official CAMUS 4CH validation split. It identified several patients with relatively low patient-level mean Dice scores, including `patient0284`, `patient0291`, and `patient0314`.



The purpose of EXP-011 was to examine selected cases in greater detail by:



1\. Measuring the overlap and disagreement between predicted and ground-truth LV masks.

2\. Identifying whether each selected prediction showed under-segmentation or over-segmentation.

3\. Quantifying true-positive (TP), false-positive (FP), and false-negative (FN) pixels.

4\. Comparing predicted and ground-truth foreground pixel counts.

5\. Generating visualizations showing the original ultrasound, ground-truth mask, prediction, overlays, and spatial error distribution.

6\. Comparing the two cardiac phases of `patient0284`.



This was a focused failure-analysis experiment using the existing checkpoint. No retraining, fine-tuning, threshold optimization, or preprocessing modification was performed.



## 2. Dataset and Case Selection



The experiment used the official CAMUS validation split and the 4CH view only.



Four patient-phase combinations were investigated:



| Patient | View | Phase |

|---|---|---|

| `patient0291` | 4CH | ES |

| `patient0314` | 4CH | ES |

| `patient0284` | 4CH | ED |

| `patient0284` | 4CH | ES |



The cases were selected based on the difficult patient-level validation results identified in EXP-010.



`patient0291` and `patient0314` had particularly low ES Dice scores. `patient0284` had the lowest patient-level mean Dice among the 50 validation patients, making it useful to examine both its ED and ES predictions.



This was a focused, performance-informed selection rather than a random or exhaustive evaluation of all validation cases.



### 2.1 Input and ground-truth preparation



The existing CAMUS loading and preprocessing implementations were reused.



For each selected case:



- The dedicated ED or ES ultrasound frame was loaded.

- The corresponding ground-truth annotation was loaded.

- The CAMUS annotation was converted into a binary LV cavity mask.

- The original image and mask were processed using the existing preprocessing pipeline.

- The model input and ground-truth mask were represented on the 256 × 256 model-input grid.



The binary mask represents:



- `0`: Background

- `1`: LV cavity



The original annotation was transformed through the same preprocessing pipeline used for the model input.



Before inference, the experiment checked that the transformed original annotation matched the ground-truth mask returned by the dataset implementation. Geometry metadata was also compared.



**Both correspondence and geometry checks passed for all four cases.**



No new preprocessing transformation was introduced.



## 3. Model and Inference Configuration



The existing U-Net checkpoint was used without retraining or modification.



| Configuration | Value |

|---|---|

| Architecture | 2D U-Net |

| Input channels | 1 |

| Output channels | 1 |

| Parameters | 1,942,289 |

| Checkpoint | `unet_4ch_best_full_validation.pth` |

| Device | CPU |

| Input dimensions | 256 × 256 |

| Output representation | Logits |

| Probability conversion | Sigmoid |

| Probability threshold | 0.50 |

| Prediction post-processing | None |

| Evaluation split | Official validation |



The model produces logits, which are converted into probabilities using sigmoid.



The binary prediction is generated using:



```python

prediction = probability >= 0.50

```



The experiment uses raw binary predictions.



No largest-connected-component filtering, morphological operation, or other post-processing was applied.



The same checkpoint, preprocessing pipeline, and global threshold were retained across all four cases.



**Checkpoint training-state caution:** The existing checkpoint represents a partially trained baseline rather than a demonstrated fully trained or converged model. The results must be interpreted in that context.



## 4. Failure Analysis Method



The analysis compares the predicted binary LV mask with the corresponding binary ground-truth mask.



For each pixel, the comparison identifies:



- **True Positive (TP):** Pixel correctly predicted as LV.

- **False Positive (FP):** Pixel predicted as LV but not present in the ground truth.

- **False Negative (FN):** Ground-truth LV pixel missed by the prediction.



Dice and Intersection over Union (IoU) are calculated as:



\\\[

Dice = \\frac{2TP}{2TP+FP+FN}

\\]



\\\[

IoU = \\frac{TP}{TP+FP+FN}

\\]



The relative foreground area difference is calculated using model-grid pixel counts:



\\\[

\\text{Relative Area Difference} =

\\frac{\\text{Predicted LV Pixels}-\\text{Ground-Truth LV Pixels}}

{\\text{Ground-Truth LV Pixels}}\\times100

\\]



A negative value indicates that the predicted foreground contains fewer pixels than the ground truth. A positive value indicates that it contains more.



These are pixel-based comparisons on the 256 × 256 model-input grid. They are not physical area or volume measurements.



### 4.1 Visualization method



A six-panel figure was generated for each selected case.



The figure uses a 2-row × 3-column layout:



| Panel | Content |

|---|---|

| 1 | Original-resolution grayscale ultrasound image |

| 2 | Binary ground-truth LV mask |

| 3 | Raw predicted binary LV mask |

| 4 | Model input with ground-truth LV overlay |

| 5 | Model input with predicted LV overlay |

| 6 | TP/FP/FN spatial error map |



The error map uses the following color representation:



- Green: True Positive

- Red: False Positive

- Blue: False Negative



Each figure includes the patient identifier, view, phase, prediction threshold, Dice score, and IoU score.



The figures were generated at 15 × 10 inches and saved at 200 DPI.



The original ultrasound panel uses the original image resolution, whereas the mask, overlay, and error-map panels use the 256 × 256 model-input grid.



These visualizations provide a structured representation of the segmentation and its spatial disagreement. Their generation and file existence were verified; no separate claim of independent visual review is made here.



## 5. Observed Results



The experiment produced the following measurements:



| Patient | Phase | GT Pixels | Predicted Pixels | TP | FP | FN | Dice | IoU | Relative Area Difference |

|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

| patient0291 | ES | 3,793 | 2,045 | 1,969 | 76 | 1,824 | 0.674546 | 0.508917 | -46.08% |

| patient0314 | ES | 2,433 | 2,919 | 1,746 | 1,173 | 687 | 0.652466 | 0.484193 | +19.98% |

| patient0284 | ED | 6,072 | 7,221 | 5,047 | 2,174 | 1,025 | 0.759347 | 0.612054 | +18.92% |

| patient0284 | ES | 5,111 | 6,698 | 4,372 | 2,326 | 739 | 0.740452 | 0.587871 | +31.05% |



All measurements are calculated from the raw binary predictions at threshold 0.50.



### 5.1 patient0291 — ES



Observed results:



- Ground-truth LV pixels: 3,793

- Predicted LV pixels: 2,045

- True positives: 1,969

- False positives: 76

- False negatives: 1,824

- Dice: 0.674546

- IoU: 0.508917

- Relative area difference: -46.08%



False-negative pixels substantially outnumber false-positive pixels.



The predicted foreground mask contains considerably fewer pixels than the ground-truth mask. This indicates a clear under-segmentation pattern for this case.



### 5.2 patient0314 — ES



Observed results:



- Ground-truth LV pixels: 2,433

- Predicted LV pixels: 2,919

- True positives: 1,746

- False positives: 1,173

- False negatives: 687

- Dice: 0.652466

- IoU: 0.484193

- Relative area difference: +19.98%



False-positive pixels outnumber false-negative pixels.



The predicted foreground contains more pixels than the ground truth, indicating an over-segmentation pattern for this case.



### 5.3 patient0284 — ED



Observed results:



- Ground-truth LV pixels: 6,072

- Predicted LV pixels: 7,221

- True positives: 5,047

- False positives: 2,174

- False negatives: 1,025

- Dice: 0.759347

- IoU: 0.612054

- Relative area difference: +18.92%



False-positive pixels outnumber false-negative pixels.



The predicted foreground contains more pixels than the ground truth, indicating an over-segmentation pattern.



### 5.4 patient0284 — ES



Observed results:



- Ground-truth LV pixels: 5,111

- Predicted LV pixels: 6,698

- True positives: 4,372

- False positives: 2,326

- False negatives: 739

- Dice: 0.740452

- IoU: 0.587871

- Relative area difference: +31.05%



False-positive pixels substantially outnumber false-negative pixels.



The predicted foreground contains more pixels than the ground truth, indicating over-segmentation.



## 6. Qualitative Observations



The four cases demonstrate different forms of disagreement between the predicted and ground-truth LV masks.



| Case | Dominant Error Pattern | Observation |

|---|---|---|

| patient0291 ES | Under-segmentation | False negatives substantially exceed false positives. |

| patient0314 ES | Over-segmentation | False positives exceed false negatives. |

| patient0284 ED | Over-segmentation | Predicted foreground is larger than the ground truth. |

| patient0284 ES | Over-segmentation | Predicted foreground is substantially larger than the ground truth. |



The results show that difficult segmentation cases do not necessarily share the same error pattern.



`patient0291 ES` has a substantially smaller predicted foreground mask, whereas the other three cases have larger predicted foreground masks.



### 6.1 ED–ES comparison for patient0284



Both cardiac phases of `patient0284` were evaluated using the same checkpoint, preprocessing, and threshold.



| Metric | ED | ES |

|---|---:|---:|

| Ground-truth LV pixels | 6,072 | 5,111 |

| Predicted LV pixels | 7,221 | 6,698 |

| True positives | 5,047 | 4,372 |

| False positives | 2,174 | 2,326 |

| False negatives | 1,025 | 739 |

| Dice | 0.759347 | 0.740452 |

| IoU | 0.612054 | 0.587871 |

| Relative area difference | +18.92% | +31.05% |



The ED prediction achieved a higher Dice score than the ES prediction.



The relative foreground area difference was also larger during ES.



Both phases showed an over-segmentation pattern, with false-positive pixels exceeding false-negative pixels.



These observations describe the two evaluated frames. They do not establish the underlying cause of the difference.



## 7. Relationship to EXP-009 and EXP-010



EXP-009 investigated probability threshold sensitivity using the official CAMUS 4CH validation split.



It found that threshold 0.50 achieved the highest overall mean Dice and IoU among the five tested thresholds. Consequently, 0.50 was retained as the global baseline threshold.



EXP-010 investigated patient-level performance using the same validation population. It identified `patient0284`, `patient0291`, and `patient0314` among the five lowest-performing patients.



EXP-011 used these earlier findings to select cases for more focused investigation.



The results are consistent with the earlier identification of these patients as difficult segmentation cases:



- `patient0291 ES` showed a Dice score of approximately 0.6745.

- `patient0314 ES` showed a Dice score of approximately 0.6525.

- `patient0284` showed relatively low Dice scores in both ED and ES.



EXP-011 does not change the threshold decision made in EXP-009 or the patient-level findings of EXP-010.



It adds a more detailed description of pixel-level disagreement in four selected cases.



## 8. Limitations



The findings must be interpreted within the following limitations:



- Only four selected patient-phase combinations were investigated.

- The cases were selected based on previous validation performance and are not a random or exhaustive sample.

- Only the CAMUS 4CH view and binary LV cavity segmentation task were evaluated.

- Only dedicated ED and ES frames were considered.

- The analysis used one existing, partially trained U-Net checkpoint.

- The checkpoint has not been demonstrated to be fully trained or converged.

- Only the global probability threshold of 0.50 was used.

- No threshold optimization or post-processing comparison was conducted in this experiment.

- No retraining or fine-tuning was performed.

- No causal explanation for the observed segmentation errors was established.

- Pixel counts and relative foreground area differences refer to the 256 × 256 model-input grid.

- Physical-coordinate restoration, calibrated cardiac dimensions, and LV volume estimation were not evaluated.

- No test-set evaluation was performed.

- Dice and IoU measure segmentation overlap and do not independently establish anatomical correctness or clinical suitability.

- The experiment does not establish clinical validity or diagnostic performance.



The results should therefore be treated as a focused descriptive investigation of selected validation cases.



## 9. Reproducibility and Generated Artifacts



**Analysis script:**



`scripts/analyze_failure_cases.py`



**Checkpoint:**



`outputs/checkpoints/unet_4ch_best_full_validation.pth`



**Output directory:**



`outputs/evaluations/exp011_failure_cases/`



### 9.1 Generated figures



Four six-panel visualization figures were generated:



| Patient | Phase | Figure |

|---|---|---|

| patient0291 | ES | `patient0291_4CH_ES_failure_analysis.png` |

| patient0314 | ES | `patient0314_4CH_ES_failure_analysis.png` |

| patient0284 | ED | `patient0284_4CH_ED_failure_analysis.png` |

| patient0284 | ES | `patient0284_4CH_ES_failure_analysis.png` |



### 9.2 Generated CSV files



Four per-case metric files were generated:



| Patient | Phase | CSV |

|---|---|---|

| patient0291 | ES | `failure_case_metrics.csv` |

| patient0314 | ES | `patient0314_4CH_ES_failure_metrics.csv` |

| patient0284 | ED | `patient0284_4CH_ED_failure_metrics.csv` |

| patient0284 | ES | `patient0284_4CH_ES_failure_metrics.csv` |



The generated files were verified to exist and contain data. The four CSV files were individually inspected, and their reported metrics matched the experiment log.



### 9.3 Execution procedure



The script is configured to process one patient-phase case per run.



From the repository root, with the project's virtual environment activated:



```cmd

set PYTHONPATH=src

python scripts\\analyze_failure_cases.py

```



Before each additional run, update the `PATIENT_ID`, `VIEW`, and `PHASE` configuration in the script to select the desired case.



The script uses the existing checkpoint and preprocessing implementation. It checks annotation correspondence and geometry metadata before performing inference.



The output handling prevents overwriting existing artifacts. Preserve the existing output files when reproducing the analysis.



The experiment's four cases were evaluated individually, not through an automatic loop.



## 10. Conclusion



EXP-011 completed a focused qualitative and quantitative investigation of four difficult LV segmentation cases from the official CAMUS 4CH validation split.



The analysis identified two distinct error patterns:



- `patient0291 ES` showed substantial under-segmentation, with 1,824 false-negative pixels compared with 76 false-positive pixels.

- `patient0314 ES`, `patient0284 ED`, and `patient0284 ES` showed over-segmentation, with false-positive pixels exceeding false-negative pixels.



The comparison of `patient0284` across ED and ES showed a lower Dice score and larger relative foreground area difference during ES.



The experiment also generated four six-panel visualization figures and four per-case CSV metric files.



These findings provide a more detailed description of the existing model's behavior on selected difficult validation examples. They do not establish the causes of the observed errors, demonstrate a generalizable improvement, or prove that the model is anatomically or clinically reliable.



**Status: Completed — validation-only qualitative and quantitative failure-case analysis.**



---



**Research-use statement:** This project is an educational/research proof-of-concept. It is not a clinically validated diagnostic system and must not be used to make clinical decisions.

