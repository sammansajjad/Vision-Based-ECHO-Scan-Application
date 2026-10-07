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

