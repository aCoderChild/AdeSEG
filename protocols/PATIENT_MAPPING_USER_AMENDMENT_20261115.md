# Patient mapping amendment — user-provided, 15 November 2026

This amendment records new information from the project owner. It does not alter frozen study plans, prior receipts, predictions, labels or scores. Earlier reports saying patient identity was unknown accurately described the evidence available when they were written; read them together with this amendment.

## User-provided mapping and scope

The user stated that “each sequence is a patient” and then explicitly confirmed: “Each sequence is a different patient.” For analysis, apply that statement to the distinct positive-video sequence IDs `seq1`–`seq23` in the project's `data/polypgen_sequence.jsonl`: different sequence IDs are different patients. This is a **user-reported dataset mapping**, not independently verified patient/examination metadata or an institution-issued mapping. The manifest itself still has no explicit patient identifier. We have not established how negative-only sequences, individual still frames, or any other dataset map to these patients; do not extend the statement to those sources without further evidence.

## Effect on the current studies

| Study partition | Sequence IDs | Interpretation under the user's mapping |
|---|---|---|
| Current source-group refiner training | seq2, seq3, seq4, seq5, seq12, seq13, seq14 | Seven distinct patients, per the user |
| Current fixed-read development validation | seq9, seq15 | Two distinct patients, each distinct from the seven training patients, per the user |
| Historically difficult cases held out of this refiner training | seq8, seq11, seq20, seq23 | Four distinct patients, each distinct from the above training patients, per the user |
| Completed full-video factorial | seq11, seq20, seq23 | Three distinct patients, per the user; the model was not trained on these sequence IDs |

This supports describing the *sequence-based refiner training and validation split* as patient-disjoint **under the user-provided mapping**. It does not show that every model component is patient-disjoint: the original YOLO detector's precise training split, still-frame-to-patient mapping, SAM2.1 pretraining exposure and other checkpoint exposure remain unresolved. The earlier mixed CPU/MPS pilot similarly used distinct sequence IDs for train and validation, but that does not establish independence of detector training data or provide unseen validation outcomes.

## What does not change

- `seq9` and `seq15` have already been inspected and repeatedly used to guide development; patient separation does not make them blinded or independently selected validation. The three full-video cases `seq11`, `seq20`, and `seq23` and other C6 results were likewise inspected and used in development. Do not label those outcomes a locked or untouched test set.
- The prior source-group preflight reported zero exact-byte and decoded-RGB duplicates across its screened train, validation and priority/C6 image groups. That check did not assess near-duplicates, all outside still images, original video sourcing or detector pretraining exposure. Shared filename prefixes—especially `seq6`/`seq10` and `seq3`/`seq4`—do not by themselves prove common patient or video, but the mapping does not explain their source naming. Retain these as data-provenance checks rather than assuming either a shared patient or an error in the user's statement.
- The seq23 inset-exclusion viewport was chosen retrospectively; the user mapping does not make its selection prospective or automatically applicable to other layouts. Continue to use the exact same viewport for matched comparisons and disclose it.
- Frames from one sequence remain correlated: they are not separate patients. Patient/sequence is the intended unit of aggregation or uncertainty, but two development patients and three selected video patients do not support precise population inference or an absence-safety claim. Labeled-empty frames need not be clinically confirmed absence.
- Frozen model selection, genuinely uninspected patients/centres, an explicit outcome and false-alarm margin, near-duplicate and source audits, detector and checkpoint exposure, and applicable data permissions remain needed for the planned confirmatory evaluation. A distinct sequence ID is not, by itself, proof that all training components have never seen that patient.

## Provenance

Project-owner confirmation arrived in this conversation on 15 November 2026. No patient names or identifiers were collected. No validation/test masks were opened to produce this amendment. The relevant unchanged source files are `protocols/source_group_capture_v1.json`, `protocols/SOURCE_GROUP_CAPTURE_FEASIBILITY_20261114.md`, `protocols/MICCAI_2027_EXPERIMENT_PROTOCOL.md`, and the completed preflight `source_group_capture_preflight_attempt03/manifest.json` under the user's Drive experiment folder.
