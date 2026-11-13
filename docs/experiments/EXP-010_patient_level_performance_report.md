# EXP-010 — Patient-Level ED/ES Segmentation Performance Analysis



## 1. Objective



Investigate patient-level variation in left-ventricular (LV) cavity segmentation performance and compare model performance between end-diastole (ED) and end-systole (ES).



The experiment evaluates whether segmentation performance is consistent across patients and whether one cardiac phase demonstrates higher Dice scores than the other.



This is a descriptive validation experiment. It does not attempt to establish statistical significance or clinical validity.



## 2. Dataset and Evaluation Protocol



**Dataset:** CAMUS public dataset

**Split:** Official validation split

**View:** 4CH

**Target:** Binary LV cavity segmentation

**Phases:** ED and ES

**Patients:** 50

**Frames:** 100 (50 ED and 50 ES)



The official patient-level validation split was used. Each patient contributed one ED frame and one ES frame.



The test split was not used.



### 2.1 Input preparation



The existing `CAMUSDataset` implementation was reused.



- Target input dimensions: 256 × 256.

- Aspect-ratio-preserving resizing and centered zero-padding.

- Linear interpolation for images.

- Nearest-neighbor interpolation for masks.

- Image intensity conversion to float32 and division by 255.

- Binary LV cavity masks generated through the existing CAMUS mask conversion function.



No new preprocessing transformation was introduced.



## 3. Model and Inference Configuration



The existing U-Net checkpoint was evaluated without retraining.



| Configuration | Value |

|---|---|

| Architecture | 2D U-Net |

| Input channels | 1 |

| Output channels | 1 |

| Parameters | 1,942,289 |

| Checkpoint | `outputs/checkpoints/unet_4ch_best_full_validation.pth` |

| Device | CPU |

| Output activation | Sigmoid |

| Probability threshold | 0.50 |

| Prediction post-processing | None |

| Evaluation metric | Dice coefficient |



Binary predictions were generated using:



`prediction = probability >= 0.50`



No largest-component filtering or other post-processing was applied.



The same model architecture, checkpoint, input preparation, inference approach, and Dice calculation conventions used by the existing validation evaluator were retained.



## 4. Patient-Level Pairing Method



Each evaluated sample provided:



- `patient_id`

- `phase`

- `image`

- `mask`



ED and ES measurements were paired using the patient identifier and phase label, rather than relying on dataset indices or frame ordering.



For each patient:



**Patient mean Dice**



`mean_dice = (ED_dice + ES_dice) / 2`



**ED–ES difference**



`ED_minus_ES = ED_dice - ES_dice`



Interpretation:



- Positive difference: ED Dice is higher.

- Negative difference: ES Dice is higher.

- Zero difference: Both phases have equal Dice.



The script checked the following integrity conditions:



- Exactly 50 unique validation patients.

- Exactly 100 evaluated frames.

- Exactly 50 ED frames.

- Exactly 50 ES frames.

- No duplicate patient-phase pairs.

- No missing ED or ES phases.

- Exactly one ED/ES pair per patient.



All checks passed.



## 5. Observed Results



### 5.1 Overall phase-level performance



| Phase | Mean Dice |

|---|---:|

| ED | 0.890374 |

| ES | 0.853143 |



The mean ED Dice was higher than the mean ES Dice by 0.037231.



### 5.2 Patient-level performance statistics



| Statistic | Patient mean Dice |

|---|---:|

| Mean | 0.871759 |

| Median | 0.882385 |

| Standard deviation (population) | 0.046743 |

| Minimum | 0.749900 |

| Maximum | 0.934819 |



The patient-level standard deviation was calculated using population standard deviation (`ddof=0`) across the 50 validation patients.



### 5.3 ED–ES paired differences



| Statistic | Value |

|---|---:|

| Mean ED–ES difference | +0.037231 |

| Median ED–ES difference | +0.030481 |

| Patients with higher ED Dice | 35 |

| Patients with higher ES Dice | 15 |

| Exact ties | 0 |



ED Dice was higher for 35 of the 50 patients (70%), while ES Dice was higher for 15 patients (30%).



These counts describe the observed validation patients and do not establish a statistically significant phase effect.



### 5.4 Five lowest-performing patients



Patients are ranked by their mean ED/ES Dice.



| Patient | ED Dice | ES Dice | Mean Dice |

|---|---:|---:|---:|

| patient0284 | 0.7593 | 0.7405 | 0.7499 |

| patient0291 | 0.8517 | 0.6745 | 0.7631 |

| patient0314 | 0.8763 | 0.6525 | 0.7644 |

| patient0232 | 0.8302 | 0.7310 | 0.7806 |

| patient0268 | 0.8419 | 0.7608 | 0.8013 |



### 5.5 Five highest-performing patients



| Patient | ED Dice | ES Dice | Mean Dice |

|---|---:|---:|---:|

| patient0279 | 0.9394 | 0.9302 | 0.9348 |

| patient0265 | 0.9192 | 0.9468 | 0.9330 |

| patient0303 | 0.9151 | 0.9474 | 0.9312 |

| patient0271 | 0.9541 | 0.9035 | 0.9288 |

| patient0256 | 0.9237 | 0.9328 | 0.9283 |



## 6. Interpretation



The model achieved a mean patient-level Dice score of 0.871759 across the 50 validation patients.



The phase-level means indicate higher segmentation overlap during ED (0.890374) than ES (0.853143).



The paired patient-level comparison supports the same descriptive observation: ED Dice was higher for 35 patients, whereas ES Dice was higher for 15.



Performance was not uniform across patients. Patient-level mean Dice ranged from 0.749900 to 0.934819.



Two particularly weak ES cases were:



- `patient0291`: ES Dice 0.6745.

- `patient0314`: ES Dice 0.6525.



Patient `patient0284` had relatively low Dice in both phases, with ED Dice 0.7593 and ES Dice 0.7405.



These patients provide useful candidates for subsequent qualitative error inspection.



The mean patient-level Dice equals the mean of the 100 frame-level Dice values because every patient contributes exactly two observations, one from each phase.



The observed ED–ES difference is descriptive. No inferential statistical test was conducted.



## 7. Relationship to EXP-009



EXP-009 investigated probability threshold sensitivity on the same 100-frame validation set.



The global threshold of 0.50 achieved the highest overall mean Dice and was retained as the baseline.



EXP-010 therefore used the same global threshold rather than introducing phase-specific thresholds.



The patient-level results provide additional detail about variation hidden by an overall mean. In particular, `patient0291` remains a difficult ES case, consistent with its previously observed weak segmentation performance.



EXP-010 does not change the threshold decision made in EXP-009.



## 8. Limitations



- Evaluation was restricted to 50 official validation patients from the CAMUS 4CH view.

- Only ED and ES frames were evaluated.

- Results depend on the existing checkpoint and its training state.

- The checkpoint represents partial training rather than a fully trained or demonstrated-converged model.

- No confidence intervals or inferential statistical tests were calculated.

- No correlation analysis with expert image quality, LV size, or LV shape was performed.

- No causal explanation for the ED–ES performance difference was established.

- No physical-coordinate measurements or volume estimates were evaluated.

- No final test-set evaluation was performed.

- Dice measures segmentation overlap and does not establish anatomical correctness, diagnostic performance, or clinical suitability.



## 9. Reproducibility



**Analysis script:**



`scripts/analyze_patient_performance.py`



**Checkpoint:**



`outputs/checkpoints/unet_4ch_best_full_validation.pth`



**Exported patient-level results:**



`outputs/evaluations/exp010_patient_performance.csv`



The CSV contains one row per patient and the following fields:



- `patient_id`

- `ED_dice`

- `ES_dice`

- `mean_dice`

- `ED_minus_ES`



The CSV was verified to contain exactly 50 rows and 50 unique patient IDs.



To reproduce the experiment from the project root:



```cmd

python scripts\\analyze_patient_performance.py

```



The script refuses to overwrite an existing CSV with the same filename.



## 10. Conclusion



EXP-010 successfully completed patient-level analysis of the existing U-Net on the official CAMUS 4CH validation split.



The mean patient-level Dice was 0.871759. ED had a higher mean Dice than ES, and ED Dice was higher for 35 of the 50 patients.



The results demonstrate patient-level variation and identify several difficult segmentation cases, particularly during ES.



The experiment provides a descriptive basis for subsequent qualitative failure analysis. It does not establish statistical significance, generalization to unseen test patients, or clinical validity.



**Status: Completed — validation-only descriptive analysis.**
