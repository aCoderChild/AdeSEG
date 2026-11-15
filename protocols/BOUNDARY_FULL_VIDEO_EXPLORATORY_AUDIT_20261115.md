# Boundary refiner full-video factorial: independent design audit

Status: design and code review, not an execution receipt. The user explicitly requested the full-video A/B/C/D comparison after the exploratory fixed-read results. This is an explicitly user-requested exploratory protocol amendment; it is not a passed predeclared promotion gate, a blinded test, or evidence of clinical effectiveness. No ground-truth mask is used to initialize or refine any evaluation frame.

## Tested checkpoint and scope

The boundary-factorial `region_boundary` checkpoint uses `ForegroundConditionedRefiner(gate_mode='region', contrast_head=True)`, 48,005 trained parameters, 1,640 training steps, source-bound MPS features, the corrected CMSA-inspired weighted objective, one seed and the 214-read training/development manifest. Its backbone SHA begins `a2345a`, its feature-pipeline SHA begins `bea5e5`, and its actual manifest SHA begins `827f6b`. Require complete actual hashes and validate all frozen plan, replay and independent-audit receipts, not these prefixes.

The separately trained same-objective `no_contrast_boundary` control has `contrast_head=False` and narrowly higher fixed-read score on each of the two inspected validation sequences. It must not be loaded by changing checkpoint metadata or represented by the contrast-enabled constructor. A region-only video test cannot establish added contrast value.

The fixed-read study used two previously inspected development videos, one seed and three labeled-empty frames from one video. Region/boundary mean present Dice was 0.9005 versus 0.8970 native on 47 present reads, while no-contrast/boundary was 0.9014. Each method retained one zero-overlap case and 0/3 labeled-empty false alarms. The boundary loss changes both weighting and an IoU term, so it does not isolate either component.

## Routing and conditional causal interpretation

| Arm | Returned logits | Mask sent to memory encoder | Relevant controlled comparison |
|---|---|---|---|
| A | original | original | shared matched baseline |
| B | refined | original | B vs A: output-only, same native memory trajectory |
| C | original | refined | C vs A: effect of memory writes on future frames |
| D | refined | refined | D vs B: effect of memory writes with refined output |

The conditioning mask and its associated memory are protected. When enabled, the existing adapter changes the low-resolution tensor returned to propagation and/or the high-resolution mask input to `_encode_memory_in_output`. It leaves the native compact `pred_masks`, object score, object pointer and any pruning rules intact. In particular, selection code that consumes the stored native mask continues to see the native mask even when the memory encoder consumes the refined one. C/D therefore alter the encoder's mask input, not every mask-derived component of the predictor. Exact next-frame decoder outputs in C/D may differ after the first changed memory write; do not demand later identity with A.

B must reproduce arm A's entire native stored compact state, window/insurance events, selected-token and positional traces frame by frame; refined returned masks may differ. A/C and B/D should share their first post-prompt native decoder output and object state before feedback can act; conditioning outputs should agree across all arms. A live route must show that a genuine loaded checkpoint, not only a synthetic/identity refiner, reaches all routing branches and obeys the expected tensor geometry and native resize.

## Suppression and pointer consistency

SAM2's `_forward_sam_heads` writes −1024 to all candidate mask logits when the native object score is non-positive. The residual refiner is applied after this suppression and is unbounded. It can therefore theoretically yield positive logits while the native object score and no-object pointer remain unchanged. When configured, the native memory encoder can additionally insert its no-object embedding according to that same unchanged score even if refined foreground is supplied. This is a consequence of preserving original predictor decisions, not proof a flip happens on these videos.

An independent label-free CPU replay of the actual checkpoint on its frozen seq4/frame24 training read found all native logits −1024, refined minimum −690.82 and maximum −190.60, with zero refined positive low-resolution pixels. This demonstrates a large change but not an observed suppression flip. During the real run, record original object score, original/refined low-resolution ranges, positive output and memory pixels, and count frames where the existing native score indicates absence while the refined result contains foreground. These are observational diagnostics only: do not gate, redetect or introduce a new inference threshold. Record pointer and no-object behavior alongside the output and memory input.

## Comparator and metric compatibility

Use the same archived automatic initialization frame/mask, SAM2.1 checkpoint, original detector lineage, exact input-image hashes, 16,384-token project window, 3,072-token insurance reserve, indexed RoPE and scoped MPS pointer fix as the completed same-device project Lean comparison. Historical YOLO evaluated successive frames until its first detection; the new video runs replay the saved automatic mask and make no detector calls. This is one initialization, not necessarily one historical detector call.

Seq23 must use and restore the same fixed inset-excluded viewport; the crop was chosen retrospectively. Exclude conditioning frames from the main post-prompt table, but separately report all-frame and pre-prompt frames. Include the exact present-frame and labeled-empty denominators and empty predicted area. Recheck source images and mask-prompt hashes, model tensor fingerprints, full frame coverage, token limits, memory events and A-to-B state identity before labels are joined.

The available same-MPS native/project Lean/SAMURAI/SAM2Long comparisons use seq11, seq20 and seq23, but the project Lean is a modified token-window model, not unmodified author Lean. SAM2Long's final backtracking is noncausal; SAMURAI retains its author's thresholds. CMSA uses a different model/checkpoint and was unprompted; the archived seq23 CMSA run used the full image including the inset. Do not label CMSA a matched-input comparison. The native comparator's shared-helper hashes were measured during its runs rather than bound before inference, which limits provenance certainty.

## Engineering boundary

The old `verified_mps_window_refiner` is not compatible with this checkpoint: it requires a missing historical `runner_source_sha256`, a different CPU feature-pipeline identity and old hard-coded 19/10 matched controls. A new version must explicitly bind the actual boundary checkpoint, manifest, capture/source-freeze/MPS environment and choice of model constructor. It must not patch the old checkpoint's metadata or falsify a screen receipt. The user's request permits an exploratory execution after engineering verification, not a claim that historical eligibility criteria were met.

Validate a learned checkpoint through protected-conditioning A/B/C/D on short seq4 and cropped seq23 clips, with two fresh process runs, the original MPS pointer/RoPE fixes, exact model/refiner parameter and buffer fingerprints, source/position/token traces, native/reference and refined output routing, memory-mask geometry and single-object behavior. Ground truth is not used in model inputs or runtime selection. Independently verify each resulting artifact and environment before attempting full videos. Initial observations from this retrospective test cannot establish patient independence or guide claims of generalization.
