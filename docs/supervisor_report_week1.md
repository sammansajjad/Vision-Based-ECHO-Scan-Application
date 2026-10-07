# Week 1 Progress Report

Project: Computer Vision Based Assistive Application for Analyzing ECHO Scans

## 1. Objective

Define the responsibilities and boundaries of the Computer Vision (CV) subsystem and the interfaces between it and the downstream components.

## 2. Work Completed

- Reviewed basic concepts: ECHO A4C/4CH view, left ventricle (LV), image vs. mask vs. label, segmentation, patient-level data splitting.
- EXP-001: inspected the 4CH image sequences and ground-truth masks of three CAMUS patients (patient0001 to patient0003). Details are in docs/experiment_log.md.
- Wrote the CV pipeline contract (docs/pipeline_contract.md): 10 sections defining the CV boundary, the data structures passed between stages, and 7 open questions.
- Created the initial code package (src/cv_pipeline) with the six data containers defined in the contract. They were checked only with small synthetic arrays.
- Committed and pushed the work to the project repository.
## 3. Findings

- The 4CH sequences of the three inspected patients contain multiple frames (20, 15 and 21 frames for patient0001 to patient0003).
- Image sizes differ between the inspected patients (549 x 389, 748 x 584 and 591 x 487), so the pipeline cannot assume one fixed input size.
- The ground-truth masks of the inspected patients contain labels 0 (background), 1 (LV cavity), 2 (LV myocardium) and 3 (left atrium).
- ED and ES are different frames, and the ES frame number is patient-specific and read from metadata (Info_4CH.cfg).
- For patient0001, the LV cavity covers 33,363 pixels at ED and 18,261 pixels at ES. These are 2D pixel counts, not physical areas or volumes.

## 4. Limitations

- Only three patients were inspected. The observations must not be generalized to the whole CAMUS dataset.
- No segmentation model has been trained or evaluated. No accuracy results exist yet.
- No pipeline stage has been run on real data. The data containers were checked only with synthetic arrays.
- Pixel counts are not calibrated physical measurements. No clinical validity is claimed.
- Support for other modalities (for example cardiac MRI) is a design intention only and has not been implemented or tested.

## 5. Open Questions

These are recorded in docs/pipeline_contract.md and are not yet decided:

- Who implements and maintains the CAMUS adapter (to be agreed with the teammate).
- How ED/ES frame labels are represented.
- How masks and measurements are mapped back to the original image size if preprocessing resizes images.
- What defines an unusable image, and how an implausible mask is detected.
- Whether the segmentation target is the LV cavity only or the cavity plus myocardium.
- Which features will be extracted, and whether spatial calibration metadata is usable.

## 6. Next Steps

- Inspect more CAMUS patients before drawing any dataset-level conclusion.
- Agree the CAMUS adapter ownership and the retry/GUI handling of result statuses with the teammate.
- Decide the segmentation target after further inspection.
- Establish a baseline segmentation experiment, with a patient-level split, before any optimization.