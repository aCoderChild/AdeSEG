# Exploratory cross-fit observer capture

This engineering design uses only the reviewed V2 index preflight and does not
claim clinical validation, independent patients, model improvement or blind
testing. The V1 index preflight is superseded because it recorded the wrong
existing-manifest digest. Seq1/7 have no initialized object and are not given
artificial captures. The existing 214 observations are reused only if their
actual manifest SHA and the runtime/pipeline relationship are independently
verified. No training or scoring labels are opened during any capture.

For each of the 12 initialized sequences, replay the archived automatic mask
in a fresh process using the existing audited project Lean MPS runtime. Keep
the original 16,384-token memory window, 3,072 insurance reserve, anchor,
presence/pointer logic and prompt schedule. Do not run the detector or use
new gates, thresholds, redetection or pruning rules. Seq23 is staged through
the exact recorded [446,0,1280,720] viewport; predictions are restored to the
full source frame for comparison. Snapshots record this viewport explicitly;
any later training loader must crop labels to the same ROI and verify alignment.

Run a fresh unobserved reference and then a separately loaded observer pass.
Require both to reproduce the frozen automatic-prompt A trajectory, including
all masks, native state hashes, memory events, position/selection traces and
all frame-input bytes. Capture only the index-plan target frames through the
existing read-only decoder observer. Rehash all source inputs before and after
each run, prohibit annotation-mask opens, check finite logits and loaded model
fingerprints, and save all outputs under the requested Drive experiment folder.

Before any full capture, run a source-bound short reference and fresh-process
observer on full-frame seq6 and cropped seq23. They
cover nonzero and zero initialization indices as well as both viewport types;
the full-reference check remains mandatory for every sequence. Check exact
state/mask/trace agreement and independently
audit the artifacts. The old 214 captures were produced after warming the
archived box/points before adding the automatic mask, whereas the A comparison
uses the saved automatic mask directly. The new observer must not silently
claim equivalence: compare overlapping seq6 observations at frames 24 and 90
against the independently audited historical MPS capture and bind both sets of
tensor hashes. Failure blocks use of the new observations as equivalent.
Seq23's observer tensors are not independently repeated; the short check is
an engineering no-op check, not a full measurement of MPS repeatability.
The user has explicitly waived CPU-versus-MPS numerical parity for future
experiments; this instruction supersedes that item in the frozen V2 index
preflight. The short checks here run on MPS and retain finite-output, prompt,
source, routing and no-op validation. CPU is used only to audit saved artifacts.
The capture receipt binds archived frame-metadata file paths and hashes but
does not parse their historical outcome fields or open annotation masks. A
later fold-scoped trainer may selectively project only the authorized label
hashes and must verify label bytes before and after reading them.

Space planning is conservative and dynamic. Before every subprocess require
22 GiB reserve plus the estimated remaining uncompressed tensor payload and
4 GiB staging/swap/log margin; check free space repeatedly during each child
and stop it if the reserve is crossed. These estimates are not a Drive quota
guarantee. The small prior zlib probe is not a validated future capture format;
this design saves raw compatible snapshots and does not delete prior records.
Do not launch any capture until the active GPU study finishes, this source is
independently reviewed, and the full headroom check succeeds.

The full stage is resumable at the sequence level. Each launch runs one
reference, one observer capture and one independent audit in a new folder;
seq6 must pass first. Every later invocation supplies all previous successful
independent audit paths using repeated `--prior-audit` options. Each prior
phase log, receipt, auditor/source identity, exact planned target list and
short-gate identity is rechecked before its sequence reduces the outstanding
payload reservation. A failed sequence can be retried in a fresh folder;
prior results and partial failures are retained. An aggregate capture manifest
is written only after all 12 sequence audits cover exactly 143 planned reads.
Only that completed aggregate is an input to cross-fit training. The end of a
capture and its audit can temporarily reserve its payload twice; this can
conservatively stop a run despite enough actual space, without weakening the
22 GiB reserve or deleting files.
