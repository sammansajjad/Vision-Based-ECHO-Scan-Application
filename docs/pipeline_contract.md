# CV Pipeline Contract

## 1. Purpose and Scope

This document defines the software boundary and data contracts of the Computer Vision subsystem of the project.

The CV subsystem is responsible for processing medical image inputs, checking image usability, performing image segmentation, optionally applying experimentally validated post-processing, extracting image-derived measurements and features, and returning structured results to the downstream application.

The primary validated use case for the initial proof of concept is:

ECHO → A4C/4CH view → Left Ventricle (LV) segmentation → LV measurements/features → downstream analysis.

The CV subsystem is designed with stable interfaces so that individual implementations can be changed without requiring changes to downstream components.

The initial implementation focuses on ECHO/A4C data from the CAMUS dataset. Support for other imaging modalities, such as cardiac MRI, is not implemented in the current scope.

## 2. Pipeline Boundary

The CV subsystem begins when an image frame is provided in the generic ImageInput representation.

The CV subsystem ends when it returns a structured CVResult containing the quality result, the final segmentation mask, and the extracted feature set.

The CV subsystem is responsible for:

- preprocessing the input image
- checking image usability
- segmenting the target structure
- optionally post-processing the segmentation
- extracting image-derived measurements and features

The CV subsystem is NOT responsible for:

- dataset-specific file loading or CAMUS file-format handling (handled by a dataset/adapter layer)
- patient-level dataset splitting
- classification or downstream analysis
- GUI or application logic
- clinical diagnosis

The downstream system consumes the CVResult and does not depend on the internal segmentation model used to produce it.

## 3. ImageInput

ImageInput is the generic representation of a single image frame provided to the CV subsystem.

| Field | Meaning |
|---|---|
| image | NumPy array containing the pixel data of one frame |
| patient_id | Identifier of the patient the image belongs to |
| modality | Imaging technology (e.g. ECHO, MRI, CT) |
| view | Acquisition view (e.g. A4C/4CH) |
| frame_index | Position of this frame within its sequence |

ImageInput does not contain dataset-specific information such as CAMUS file paths, CAMUS filenames, or `Info_4CH.cfg`. A dataset-specific adapter is responsible for converting dataset data into this representation.

ImageInput represents one frame. Whether a frame corresponds to end-diastole (ED) or end-systole (ES) is not part of this contract in its current form (see Open Questions).

## 4. Preprocessor and ProcessedImage

The Preprocessor receives an ImageInput and returns a ProcessedImage.

Its purpose is to prepare the image for the next pipeline stage.

| Field | Meaning |
|---|---|
| image | NumPy array containing the preprocessed pixel data |
| source | The ImageInput this was produced from (preserves patient_id, modality, view, frame_index) |

The exact preprocessing steps (for example resizing, intensity normalization, contrast enhancement, data type, and target resolution) are NOT fixed by this contract. They are implementation details to be determined through experiments and recorded in the experiment log.

Preprocessing prepares an image. It does not decide whether the image is usable; that is the responsibility of the QualityChecker.

The Preprocessor must not discard the identifying information of the input. The patient_id, modality, view, and frame_index must remain traceable through to the final CVResult.

## 5. QualityChecker and QualityResult

The QualityChecker receives a ProcessedImage and returns a QualityResult.

| Field | Meaning |
|---|---|
| is_usable | True if the image is considered usable for segmentation, otherwise False |
| reason | Short explanation of the decision |

If is_usable is False, the pipeline stops before segmentation. The returned CVResult contains the QualityResult, and the segmentation_mask and feature_set are None (meaning "not produced"). An empty mask or zero-valued features must NOT be used to represent a skipped stage, because they would be indistinguishable from a real result.

Downstream components must check the CVResult status (see Section 9) before reading the segmentation_mask or feature_set. When is_usable is False, the status is "unusable_image".

Quality checking determines usability. It does not modify the image.

The quality criteria (for example sharpness, contrast, or other measures) and any thresholds are NOT fixed by this contract. They must be justified by experiments and recorded in the experiment log.

## 6. Segmenter and SegmentationMask

The Segmenter receives a ProcessedImage that has passed quality checking and returns a SegmentationMask.

| Field | Meaning |
|---|---|
| mask | NumPy array with the same height and width as the ProcessedImage image; 0 = background, 1 = LV target |
| source | The ProcessedImage this mask was produced from |

The exact definition of "LV target" (for example, LV cavity only, or LV cavity plus myocardium) is NOT fixed by this contract. It is an open decision (see Open Questions). Dataset ground-truth labels (CAMUS: 0 background, 1 LV cavity, 2 LV myocardium, 3 left atrium) must be mapped explicitly to this binary representation once the decision is made.

The Segmenter interface must remain stable when the internal model changes (for example, from a U-Net to another architecture). Downstream components must not depend on model-specific details.

## 7. PostProcessor

The PostProcessor receives a SegmentationMask and returns a SegmentationMask.

Post-processing is optional. It is NOT applied automatically. A post-processing step is included in the pipeline only if experiments show that it improves segmentation compared with the baseline (the same pipeline without that step), using the same data split and the same evaluation metrics. The comparison is recorded in the experiment log.

A refined mask is still a SegmentationMask. The output follows the same contract as the input (same shape, 0 = background, 1 = LV target), so the pipeline works whether or not post-processing is used.

## 8. FeatureExtractor and FeatureSet

The FeatureExtractor receives the final SegmentationMask and returns a FeatureSet.

| Field | Meaning |
|---|---|
| features | Named numerical values calculated from the mask, each with a stated unit |
| source | The SegmentationMask this FeatureSet was produced from |

Feature extraction is not classification. The FeatureSet is consumed by the downstream classification/analysis component.

All features must state their unit. Features calculated from pixel counts (for example LV pixel area) are in pixel units. They are NOT physical measurements (such as mm^2, mL, or volume) and must not be interpreted as such. Physical measurements require spatial calibration/metadata, which is not established in the current scope.

The specific features to be extracted are NOT fixed by this contract. They will be selected and validated through experiments and recorded in the experiment log.

## 9. CVResult

CVResult is the structured output returned by the CV subsystem for one ImageInput.

| Field | Meaning |
|---|---|
| status | One of "ok", "unusable_image", "processing_error" |
| quality_result | The QualityResult, or None if the pipeline failed before quality checking completed |
| segmentation_mask | The final SegmentationMask (after optional post-processing), or None if not produced |
| feature_set | The FeatureSet, or None if not produced |
| error_message | Description of the failure when status is "processing_error", otherwise None |
| source | The ImageInput this result was produced for (preserves patient_id, modality, view, frame_index) |

Meaning of each status:

| status | Situation | segmentation_mask and feature_set |
|---|---|---|
| ok | Image passed quality checking and all stages completed | Present |
| unusable_image | The QualityChecker judged the image unusable; this is an expected outcome caused by the data | None |
| processing_error | A software stage failed after or during processing; this is an error caused by the software | None |

Downstream components must check status first, and read segmentation_mask and feature_set only if status is "ok".

"unusable_image" and "processing_error" are kept separate so that poor data and software failures can be counted, reported, and handled differently. Partial results are not returned when status is "processing_error".

An all-zero segmentation mask for an image that passed quality checking is treated as "processing_error", not "ok", because a valid A4C image is expected to contain the LV.

Downstream components must not depend on which segmentation model produced the result.

## 10. Modality Independence

The pipeline interfaces (ImageInput, ProcessedImage, QualityResult, SegmentationMask, FeatureSet, CVResult) are intended to be independent of the imaging modality. Implementations behind these interfaces may be modality-specific.

The following is a design expectation. It has not been validated, because only ECHO/A4C is implemented.

| Component | Expected for a future modality |
|---|---|
| CVResult structure and status values | Reused unchanged |
| Result and mask structures | Reused unchanged |
| Preprocessor | Modality-specific implementation expected |
| QualityChecker | Modality-specific criteria expected |
| Segmenter | New trained model expected, behind the same interface |
| Generic mask statistics (for example pixel count) | Reusable |
| LV-specific geometric features | May require adjustment |
| Physical units | Require modality-specific spatial calibration |

The current scope implements ECHO/A4C only. No other modality is implemented or tested.

### Open Questions

- Q1: Who implements and maintains the CAMUS adapter (the code that converts CAMUS files into ImageInput)? Status: to be agreed with teammate.
- Q2: How should ED/ES frame labels be represented, if at all? Options: an optional field in ImageInput, or handled outside the CV subsystem. Status: undecided.
- Q3: If preprocessing changes image size, how are masks and measurements mapped back to the original image geometry? Status: undecided; to be investigated experimentally.
- Q4: What defines an "unusable" image for CAMUS A4C (which measurable criteria and thresholds)? Status: undecided; requires inspection of CAMUS image quality labels and experiments. Do not assume thresholds.
- Q5: What is the LV target for the binary mask: LV cavity only, or LV cavity plus myocardium? Status: undecided; to be settled after further CAMUS inspection and baseline experiments.
- Q6: Which features will be extracted, and is spatial calibration metadata available and usable for physical units? Status: undecided; not yet inspected. Do not assume.
- Q7: How is an implausible (not only empty) mask detected, for example a very small or fragmented mask? Status: undecided; to be investigated experimentally. Do not assume thresholds.