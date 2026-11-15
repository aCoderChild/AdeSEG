# Exploratory note: recovering information before native absence suppression

This is a design assessment only. It changes no frozen code, prompt, model,
training plan or experiment, and establishes no effectiveness claim.

## What the present adapter can and cannot change

Lean SAM2's `_forward_sam_heads` obtains raw mask logits, mask output tokens,
and an object-presence logit from the native SAM mask decoder. If
`pred_obj_scores` is enabled, it replaces the mask with the constant `-1024`
whenever the object score is nonpositive. It projects the selected output
token to an object pointer and mixes in the native no-object pointer according
to native configuration. Its spatial memory encoder additionally uses the
same object score to add a no-object spatial embedding when that embedding is
configured. See `external/Lean-SAM2/sam2/modeling/sam2_base.py`, methods
`_forward_sam_heads` and `_encode_new_memory`.

The existing experimental routing refines the *already suppressed* low-res
mask after `_track_step`; it does not alter the object score or pointer. A
positive corrected mask may therefore coexist with an absent score, an
occluded-object pointer, and an absent spatial embedding. The encoder then
sigmoids the corrected logits, with its configured scale and bias. Changing
the logits' magnitude can saturate this sigmoid or shift its calibration. This
is a hard limitation of post-suppression recovery as a coherent memory-write
strategy, not evidence that changing the score or pointer will help.

## Two continuous residuals, with numerical implications

For x=-1024 in float32, `sinh(asinh(x)+r)` has derivative about 1024 at
r=0, crosses zero at r≈7.62462, has a zero-correction roundoff error of
about 0.000244 in a CPU check, and overflows at sufficiently large r (CPU
example r=100). `x + 2*(1+abs(x))*tanh(r)` has exact zero-correction identity
in that CPU check, derivative 2050 at r=0, crosses zero at r≈0.54866 and
saturates near +1026 as r→∞. Its large initial gradient and saturation can
also destabilize optimization and memory-mask calibration. No MPS numerical
validation or real-data efficacy test is implied by these numbers. Neither
formula is a new inference-time presence threshold; native SAM2 still has its
existing presence handling.

## Unsuppressed native values: possible source, not a demonstrated fix

The native decoder does compute the raw mask and token before suppression, so
an independently versioned observer or replay of the captured decoder inputs
with the same checkpoint could expose them. The current observer snapshot
records `_forward_sam_heads`' post-suppression seven-value return and input
features; it does not itself save the raw decoder return. A replay therefore
needs to reconstruct prompt-encoder inputs and the original mask-decoder call,
validate the chosen mask/token and checkpoint provenance, and compare the
replayed *post*-suppression output with the frozen source before interpreting
any raw-output statistics. There is no guarantee a raw mask for a predicted
absent object contains a valid polyp; it may increase false positives.

The raw projected pointer also exists before native no-object handling. Using
it alongside an unsuppressed mask on every frame is a possible uniform
architectural ablation that would remove or alter native occlusion handling;
it is not a drop-in equivalent to native Lean. Alternatively, training a
coherent representation that combines image/context information with the raw
mask, score, and token needs an explicit architecture and supervision design.
Neither option is justified merely by the current small, inspected-video
results. Avoid adding a separate detector, redetection, or a new hard
presence-gate/threshold.

Before a propagation claim, evaluate the raw and suppressed masks on matched
fixed reads, include empty frames and the pre-declared baselines, then compare
output-only against the memory-write intervention in full videos using matched
initialization, checkpoint and frame schedule. Record mask–score–pointer
consistency, decoder/no-object state, encoder inputs, false alarms, present
Dice and per-sequence results; verify gradients, finite values and exact
initial behavior on MPS. Training folds must remain patient-separated and
labels must never be used at inference. Existing seq11/20/23 observations are
development evidence, not an untouched test.
