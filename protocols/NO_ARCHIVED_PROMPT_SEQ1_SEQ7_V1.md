# Image-only automatic initialization for seq1 and seq7

This is a separately versioned, exploratory extension for the two PolypGen
sequences without archived automatic prompts. The user reports that each
sequence represents a different patient. Their previous metadata or baseline
inspection means these are not blinded or external cases. Annotation masks are
never inference inputs or prompt-selection signals.

Reuse the existing original detector checkpoint and `detect_box` implementation
(`conf=0.25`, `iou=0.7`, `imgsz=640`, class 0, CPU); no new detector threshold
or post-initialization redetection. Scan images in manifest order only until
the first valid box, exactly as the historical automatic system. Reuse the
original Lean CPU predictor, box mask and early-FPN feature-point code. When
feature points are unavailable, retain the automatic box-only mask. If a
detected box produces an empty final mask, record `detected_empty_box_only` and
stop: the historical predictor can propagate its box state, but the current
mask-initialized factorial runner cannot yet faithfully replay it. Never turn
this case into a no-detection result or a ground-truth prompt.

If there is no valid detection in the entire sequence, the historical rule
is an empty prediction for every frame. Save explicitly labeled A/B/C/D empty
prediction files for all frames; no model was initialized or refiner run in
this case. No-detection is not evidence of clinical absence. For a detected,
nonempty automatic prompt, save the image-sized provisional and refined binary
masks and image-only provenance for later separately verified video routing;
this generator does not itself run the A/B/C/D video inference.

Before generating target prompts, require two independently produced CPU
reference replays of seq6 matching each other and the archived automatic
seq6 box, confidence, points, provisional mask and final prompt mask. Bind
the actual Lean revision, tracked cleanliness, author config/bypass,
checkpoint, detector, project code, dataset manifest, image hashes, Python,
PyTorch, NumPy and Ultralytics versions. This gate uses no annotation-mask
files. Record any discrepancy and stop; do not overwrite failed artifacts.

The CPython mask-open audit guard blocks paths in the dataset manifest but is
not a security boundary against native-code I/O; inspect the imported image,
detector and predictor code as well. All experiment artifacts live under the
configured Drive outputs directory. Patient identifiers, near-duplicate
screening, detector training exposure and annotation completeness remain
unverified. The comparison is exploratory.

Before target generation, screen seq1 and seq7 against every manifest
sequence for exact-byte and decoded-RGB duplicates and bind train/validation
images to the frozen capture preflight. This does not rule out near duplicates.
The original receipt did not record the detector-helper hash or Pillow/OpenCV
versions. The new run records those values but does not claim their historical
equality; the two exact archived-prompt replays remain mandatory.
