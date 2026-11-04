# EXP-009 --- Probability Threshold Sensitivity and Difficult-Case Analysis

**Project:** Computer Vision Based Assistive Application for Analyzing
ECHO Scans\
**Experiment ID:** EXP-009\
**Status:** Completed (validation-only analysis)\
**Task:** 4CH left-ventricular (LV) cavity segmentation\
**Evaluation split:** Official CAMUS validation split\
**Checkpoint:** `outputs/checkpoints/unet_4ch_best_full_validation.pth`

------------------------------------------------------------------------

## 1. Objective

EXP-009 investigated how the probability threshold used to convert the
U-Net's probability map into a binary LV segmentation affects
segmentation performance.

The baseline rule is:

\[ `\hat{M}`{=tex}(x,y) =
```{=tex}
\begin{cases}
1 & P(LV\mid x,y) \geq 0.50\\
0 & \text{otherwise}
\end{cases}
```
\]

The work had four goals:

1.  Confirm that the current evaluation script reproduces the
    checkpoint's stored validation metrics.
2.  Inspect the model's weakest validation example to understand the
    type of segmentation error.
3.  Measure overall validation performance at several probability
    thresholds.
4.  Compare alternative thresholds against 0.50 on the exact same
    frames, including separate ED and ES summaries.

This was an evaluation and analysis experiment. No model retraining,
checkpoint modification, or test-set evaluation was performed.

## 2. Experimental setup

  Item                                      Configuration
  ----------------------------------------- -------------------------------------
  Dataset                                   CAMUS
  Split                                     Official validation
  View                                      4CH
  Target structure                          LV cavity
  Mask                                      Binary: background 0, LV 1
  Phases                                    ED and ES
  Validation frames                         100 (50 ED + 50 ES)
  Model                                     U-Net
  Input/output channels                     1 / 1
  Parameters                                1,942,289
  Checkpoint                                `unet_4ch_best_full_validation.pth`
  Device                                    CPU
  Baseline threshold                        0.50
  Thresholds tested                         0.30, 0.40, 0.50, 0.60, 0.70
  Post-processing in threshold comparison   None
  Split used for analysis                   Validation only

The model produces logits. The evaluation converts logits to
probabilities using sigmoid, then creates the binary mask by
thresholding the probability map.

The threshold experiments use the same checkpoint, validation examples,
ground-truth masks, and evaluation metrics. Only the threshold changes.

## 3. Evaluation metrics

For binary ground truth (G) and predicted mask (P):

\[ Dice = `\frac{2|G\cap P|}{|G|+|P|}`{=tex} \]

\[ IoU = `\frac{|G\cap P|}{|G\cup P|}`{=tex} \]

Mean Dice and mean IoU are the arithmetic means of per-frame scores. For
paired comparisons, the change for a frame is:

\[ `\Delta `{=tex}Dice_t = Dice_t - Dice\_{0.50} \]

A positive value means that threshold (t) improved that frame relative
to 0.50; a negative value means it made the frame worse.

## 4. Evaluation-script check

The existing `scripts/evaluate_camus_split.py` was used to evaluate the
validation split. It loads the best checkpoint, creates the 4CH ED/ES
validation dataset, predicts the LV cavity mask, and reports raw and
largest-component post-processed Dice.

Observed output:

``` text
Split: validation | samples: 100
Mean Dice (raw): 0.872
Mean Dice (post-processed): 0.882
Median Dice (post): 0.895
Std (post): 0.054
Min / Max (post): 0.675 / 0.954
Mean Dice ED (post): 0.897
Mean Dice ES (post): 0.866
```

The raw mean Dice of 0.872 agrees with the checkpoint metadata when
rounded to three decimal places. The post-processed score is a separate
result and must not be confused with the raw checkpoint metric.

### Post-processing observation

Largest-connected-component post-processing increased the mean Dice in
this validation run from approximately 0.872 to 0.882. This is an
observed result for this evaluation run, not proof that post-processing
will improve performance on other splits or datasets.

The threshold sensitivity experiments below deliberately use **raw
predictions without post-processing**, so the effect of changing the
threshold is not mixed with the effect of connected-component filtering.

## 5. Difficult-case inspection --- patient0291 ES

The weakest post-processed validation frame in the split evaluation was:

-   Patient: `patient0291`
-   Phase: `ES`
-   Raw Dice: approximately `0.675`
-   Post-processed Dice: approximately `0.675`
-   IoU: approximately `0.509`

The visualization was generated with:

``` bat
python scripts\visualize_camus_frame.py patient0291 ES
```

Observed measurements from the inspection:

  Measurement                                                  Value
  ------------------------------------ -----------------------------
  Ground-truth LV pixels                                       3,793
  Raw predicted LV pixels                                      2,045
  Post-processed predicted LV pixels                           2,045
  Probability minimum                                          0.052
  Probability maximum                                          0.925
  Probability mean                                             0.151
  Ground-truth centroid                               (108.4, 130.6)
  Predicted centroid                                  (118.2, 125.1)
  Centroid displacement                  dx = +9.8, dy = -5.5 pixels

The predicted mask covered about 53.9% of the ground-truth mask area in
model-input coordinates.

### Interpretation

The model detected an LV region in approximately the correct area, but
the predicted region was substantially smaller than the ground truth and
was displaced from its centroid. The contour was mostly inside the
ground-truth contour, indicating an under-segmentation pattern for this
example.

The unchanged pixel count and Dice before and after largest-component
post-processing show that this operation did not alter this particular
prediction.

These observations describe one difficult frame. They do not establish
why the model under-segmented it, and they do not demonstrate a general
failure pattern across all patients.

**Coordinate caution:** These measurements are in the 256×256
model-input coordinate space. The inverse geometry restoration and
physical calibration have not been validated. Pixel counts and centroid
coordinates here must not be interpreted as physical cardiac dimensions,
areas, or volumes.

## 6. Threshold sensitivity --- all validation frames

The validation set was evaluated at five thresholds, without
post-processing.

### Results

    Threshold    Mean Dice     Mean IoU
  ----------- ------------ ------------
         0.30       0.8207       0.7046
         0.40       0.8639       0.7658
     **0.50**   **0.8718**   **0.7775**
         0.60       0.8676       0.7711
         0.70       0.8516       0.7476

### Interpretation

Threshold 0.50 achieved the highest overall mean Dice and mean IoU among
the five tested values. Lowering the threshold to 0.30 reduced overall
performance noticeably. Raising it to 0.60 or 0.70 also reduced the
overall scores.

Therefore, **0.50 is retained as the global baseline threshold** for the
current checkpoint and evaluation protocol.

This conclusion is limited to the tested validation set and thresholds.
It is not a claim that 0.50 is universally optimal for other models,
datasets, views, or acquisition settings.

## 7. Threshold response by cardiac phase

### Mean Dice

    Threshold   ED mean Dice (50 frames)   ES mean Dice (50 frames)
  ----------- -------------------------- --------------------------
         0.30                     0.8634                     0.7781
         0.40                 **0.8920**                     0.8357
         0.50                     0.8904                     0.8531
         0.60                     0.8779                 **0.8572**
         0.70                     0.8565                     0.8468

### Interpretation

-   ED had its highest observed mean Dice at threshold 0.40.
-   ES had its highest observed mean Dice at threshold 0.60.
-   At the global baseline threshold of 0.50, ED mean Dice was 0.8904
    and ES mean Dice was 0.8531.

The two phases show different threshold-response patterns in this
validation sample. However, choosing separate ED and ES thresholds based
on these same results would be a tuning decision. It would require a
separately defined validation protocol and independent evaluation before
being treated as a generalizable improvement.

For the current baseline, one global threshold of 0.50 is retained.

## 8. Paired per-frame comparison against threshold 0.50

Every alternative threshold was compared with 0.50 on the same
individual frames. This gives the number of frames that improved or
worsened, rather than relying only on aggregate means.

### All validation frames (100)

  -----------------------------------------------------------------------
    Threshold    Improved       Worse   Unchanged   Mean Dice Median Dice
                                                       change      change
  ----------- ----------- ----------- ----------- ----------- -----------
         0.30          22          78           0     -0.0510     -0.0497

         0.40          41          59           0     -0.0079     -0.0059

         0.60          41          59           0     -0.0042     -0.0070

         0.70          31          69           0     -0.0201     -0.0259
  -----------------------------------------------------------------------

### ED frames (50)

    Threshold   Improved   Worse   Unchanged   Mean Dice change
  ----------- ---------- ------- ----------- ------------------
         0.30         16      34           0            -0.0270
         0.40         27      23           0            +0.0016
         0.60         12      38           0            -0.0125
         0.70         11      39           0            -0.0339

### ES frames (50)

    Threshold   Improved   Worse   Unchanged   Mean Dice change
  ----------- ---------- ------- ----------- ------------------
         0.30          6      44           0            -0.0751
         0.40         14      36           0            -0.0174
         0.60         29      21           0            +0.0041
         0.70         20      30           0            -0.0063

### Interpretation

Across all 100 frames, every alternative threshold had a negative mean
Dice change relative to 0.50. Even where an alternative threshold
improved a substantial number of individual frames, it worsened more
frames and/or produced a negative average change.

The ED and ES breakdowns show why aggregate results alone can hide
phase-specific behavior. For example, 0.40 produced a very small
positive mean change for ED, while 0.60 produced a very small positive
mean change for ES. These phase-specific changes do not outweigh the
decision to retain a single global baseline threshold for this
experiment.

No statistical significance test was performed. The counts and mean
changes are descriptive comparisons, not claims of statistical
significance.

## 9. Difficult-frame threshold response

The patient0291 ES frame was also examined at each threshold.

    Threshold     Dice      IoU
  ----------- -------- --------
         0.30   0.7840   0.6448
         0.40   0.7159   0.5575
         0.50   0.6745   0.5089
         0.60   0.6147   0.4437
         0.70   0.5264   0.3572

For this particular frame, lowering the threshold substantially improved
overlap. The 0.30 Dice was 0.7840 compared with 0.6745 at 0.50.

This is a useful example of a frame whose probability distribution and
segmentation error respond differently from the validation-set average.
However, optimizing a global threshold for one difficult frame would
harm overall validation performance. This single case is not a
sufficient reason to change the global threshold.

## 10. Overall conclusions and decisions

1.  The raw evaluation at threshold 0.50 reproduced the checkpoint's
    stored validation Dice and IoU to the reported precision.
2.  Largest-component post-processing improved the aggregate validation
    Dice in the split evaluation, but it did not change the difficult
    patient0291 ES prediction.
3.  The patient0291 ES prediction showed substantial under-segmentation
    and centroid displacement in model coordinates.
4.  Threshold 0.50 achieved the best overall mean Dice and IoU among the
    five thresholds tested.
5.  ED and ES showed different observed threshold-response patterns, but
    phase-specific threshold selection has not been established as a
    generalizable improvement.
6.  A lower threshold helped patient0291 ES, but the same lower
    threshold reduced mean performance across the full validation set.
7.  **Decision:** retain threshold 0.50 as the global baseline. Do not
    introduce phase-specific thresholds or tune the model based on this
    one difficult example.

## 11. Limitations

-   Results are based on one existing checkpoint and the official
    validation split.
-   Only the 4CH view and LV cavity were evaluated.
-   Only five thresholds were tested.
-   No confidence intervals or statistical significance tests were
    calculated.
-   The paired frame-level comparison is descriptive; ED and ES frames
    from a patient are not independent patient observations.
-   The patient0291 analysis is a single-case inspection and cannot
    establish population-wide behavior.
-   Model-space pixel measurements are not physical measurements.
-   The checkpoint was only partially trained in the historical baseline
    run; it must not be described as a fully trained or converged model.
    The checkpoint metadata records 200 optimizer steps, while the
    training script's full training loader contains 400 batches at its
    default batch size of two. The exact historical command and
    batch-size settings were not independently confirmed.
-   No test-set performance claim can be made from this experiment.

## 12. Reproducibility and related scripts

The following project scripts were used or created during this analysis:

  ---------------------------------------------------------------------------------
  Script                                        Role
  --------------------------------------------- -----------------------------------
  `scripts/evaluate_camus_split.py`             Evaluates a selected split and
                                                reports raw and post-processed Dice

  `scripts/visualize_camus_frame.py`            Visualizes an individual
                                                patient/phase prediction

  `scripts/evaluate_threshold_sensitivity.py`   Evaluates raw validation
                                                predictions at thresholds
                                                0.30--0.70

  `scripts/compare_thresholds_paired.py`        Compares each alternative threshold
                                                with 0.50 on the same frames
  ---------------------------------------------------------------------------------

Example Windows CMD commands:

``` bat
set PYTHONPATH=src
python scripts\evaluate_camus_split.py validation
python scripts\visualize_camus_frame.py patient0291 ES
python scripts\evaluate_threshold_sensitivity.py
python scripts\compare_thresholds_paired.py
```

Run commands from the repository root with the project's virtual
environment activated. These commands assume the scripts retain the
interfaces used during this experiment. If a script's arguments or
implementation have since changed, inspect the current file before
running it.

## 13. Suggested next experiment

The next planned analysis is **EXP-010 --- Patient-Level Segmentation
Performance Analysis** on the validation split.

It should pair each patient's ED and ES observations by `patient_id`,
verify that there are exactly 50 unique patients and 100 frames with one
ED and one ES per patient, and report patient-level mean Dice:

\[ D\_{patient} = `\frac{D_{ED}+D_{ES}}{2}`{=tex} \]

This is planned work, not a result of EXP-009. It should remain
validation-only, use the current checkpoint and threshold 0.50, and must
not use the test split.

------------------------------------------------------------------------

**Research-use statement:** This project is an educational/research
proof-of-concept. It is not a clinically validated diagnostic system and
must not be used to make clinical decisions.
