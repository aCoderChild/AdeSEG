# Exploratory expansion of the boundary-refiner video factorial

The user states that each numbered sequence represents a different patient. On
that user-supplied mapping, the selected sequences below are disjoint from the
current refiner's training patients (2, 3, 4, 5, 12, 13, 14) and validation
patients (9, 15). No independent patient-ID table was provided. Previous
automatic baseline outcomes for these sequences have been inspected, and
detector training exposure is unknown: this is **not a blinded final test**.

Selection is by sequence ID alone, independent of label or outcome: seq8,
seq16, seq17, seq18, seq19, seq21 and seq22. Seq8 was one of the original
four difficult sequences. The other six are the remaining C6 sequences other
than previously tested seq20 and seq23. Run all four matched A/B/C/D routes,
using the same existing region-boundary checkpoint, token window, archived
automatic prompt mask, detector lineage and native model as the completed
factorial. Do not select or modify these sequences according to scores. The
original detector was run chronologically until its first detection; this
extension replays the archived single initialization and never redetects.

All selected archived prompts are full-source-frame masks: seq23's separate,
retrospective no-inset crop is not silently applied to the new sequences.
The selected prompt and every source image hash are validated against archived
automatic run receipts. The archive's `frames.json` includes outcome fields;
the new preflight extracts only frame indices and image hashes and records the
hash and path of the original bytes. The inference runner uses the sanitized
preflight rather than decoding the archive's outcome fields. Target mask files
are protected by the existing CPython open guard; this does not establish a
native-code security boundary. The labels and historical outcomes remain
unopened by inference. Score only after all 28 videos have frozen.

Engineering checks require the existing learned-refiner short gate, plus new
independently audited A/A-repeat/B/C/D short replays on seq8 and seq17 (which
has a later automatic initialization). Preserve initial conditioning and
native pointer/object score; assert B's full stored state and token events
match A, and demonstrate actual first-frame output and write routing. Freeze
the extended source and preflight; stop on hash/state mismatch, memory guard or
disk guard. No new inference gate, pruning rule, redetection or GT prompt.

Report per-patient post-initialization Dice, overlap and labeled-empty alarms
and area, with all-frame and pre-initialization counts, and avoid claims of
independent clinical generalization or false-alarm safety. Checkpoint was
chosen after other development results, and these sequences and historical
comparators were previously inspected; exact or decoded duplicate screening
does not exclude near duplicates or unknown detector-training exposure.
