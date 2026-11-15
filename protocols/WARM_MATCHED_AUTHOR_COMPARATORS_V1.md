# Warm-matched video comparators: design and engineering scope

This is a separate exploratory track from the already running direct-mask
comparisons. On each of the 21 automatically initialized patients, use the
same archived detector box, positive and negative points, initialization frame,
conditioning mask, source images and view. For every model, first apply the
archived box, then the point pair with labels `[1, 0]` and
`clear_old_points=False`, then access the prompt-frame early backbone feature
and compute the same foreground/background prototypes used in the project warm
capture, and finally commit the saved automatic mask. Those prototypes are
diagnostic only for the author baselines: their values are not used to select
prompts or alter logits. The additional feature access may affect runtime
cache/state and must be evaluated; feature-access cache keys and descriptor
hashes are recorded per model. The newly
generated point mask is recorded only as a diagnostic; its disagreement with
the saved mask does not change the inference path. No labels, detector calls,
redetection, new confidence gate or new token-pruning rule enter inference.
Seq23 coordinates are already viewport-relative to `[446,0,1280,720]`;
predictions are restored to full-frame coordinates for scoring. Seq1/7 have no
initialization and remain separate detector outcomes.

The current native SAM2, compatibility-corrected author Lean, SAMURAI and
SAM2Long source checkouts expose `add_new_points_or_box` and `add_new_mask`.
This is source-level API evidence, not proof that the combined calls work or
produce equivalent state on MPS. SAM2Long must retain its author configuration,
full-video stage and distinct final-path return contract; its final trajectory
is offline/noncausal. SAMURAI's and Lean's own existing author fallback and
memory settings are retained and disclosed. The project Lean/refiner uses its
own previously verified warm runtime and should be evaluated with its own
checkpoint and routing controls, not relabeled as author Lean.

Before full videos, each method needs a fresh-process MPS smoke using the same
nonzero-start automatic prompt; a second cropped seq23 smoke checks view and
coordinates. Require correct object IDs, finite box/point/mask outputs and
propagated logits, exact frame coverage, stable model fingerprint, source and
prompt hashes and zero target-label open attempts. SAM2Long also needs an
explicit test of its complete forward pathway return and original author
parameters; partial/reverse modes are not assumed valid. This is not a
CPU/MPS equivalence requirement. Freeze all requested trajectories and audit
their masks and input identities before accessing evaluation labels; report
post-prompt present Dice, empty-frame alarms, pre-prompt outcomes, patient and
frame weighted summaries. Treat predictions as different model trajectories:
matching prompt *inputs* does not establish identical internal state.

The existing fixed-read refiner training and seq23 warm-gate history impose
additional pipeline checks for that model; no reuse or validation is implied
here. Patients and prompts have been inspected previously, so this is not a
blinded independent test. No full-run quality claim follows from an API or
smoke check. A separate model-specific runner, independent gate, score/audit
and resource-controlled launcher remain required before this plan can run.
