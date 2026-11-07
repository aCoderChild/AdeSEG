# Next strategy review if Run174 fails

**Status (2026-11-07):** exploratory design note, not a frozen protocol. Run174
is still active. Do not alter or interrupt its arms because of interim scores.
The completed single-regional validation arm is below native SAM2; the generic
iterative arm has also shown a large validation regression. The iterative-
regional arm and final test are not yet available.

The candidate's isolated writer and pre-hook scaffold are implemented in
`modeling/memsam_memory_writer.py`; its two CPU unit checks pass. This is only
synthetic engineering evidence. No real-frame gate, training, or efficacy
inference has been run for this candidate. Integration review found SAM2.1
passes `20 * sigmoid(mask_logits) - 10` to its memory encoder; the writer now
recovers the soft probability cue by inverting that configured transform and
leaves the native mask input untouched. Protocol v2 records this correction and
supersedes v1 before any real-frame work.

## Candidate to consider

Test a **soft-mask-conditioned residual memory writer**. At each normal SAM2
memory write, an added small convolutional branch receives the native
`maskmem_features` and the soft probabilities recovered from the exact affine-
transformed mask tensor entering SAM2's memory encoder. It predicts an additive
correction to the stored spatial values. The
correction is initialized to zero. Every token stays in place; the native frame
schedule, positional encodings, pointers, memory attention, image encoder,
memory encoder, and mask decoder remain unchanged and frozen. No frame/token
selection, confidence threshold, or multiplicative keep/drop gate is used.

The purpose is to test whether the *content written into future memory* can be
made more useful for later segmentation. This differs from Run174's new reader,
which changes current features after the native memory read. It also differs
from token pruning/compression runs, which alter what is retrieved or retained.
Runs28–31 tested native-writeback interventions, not a trained write-side
representation correction.

## Literature basis and limits

MemSAM (CVPR 2024) uses a predicted mask to reinforce image features before
writing memory. Its public code contains a local `Conv3x3`/`Conv1x1` feature
path. It is a useful source for the insertion-point idea and module structure,
not a direct method to copy: MemSAM is based on SAMUS/XMem for echocardiography,
and its reported ablation without reinforcement loses only 0.45 mean-Dice
points. The proposed SAM2 experiment would adapt the feature path into a
zero-initialized additive residual, rather than reproduce MemSAM's
mask-multiplication output. This is a new integration hypothesis, not a claim
that the mechanism is novel or likely to win. In the upstream implementation,
the final reinforcement is the predicted mask multiplied by a channel-softmax
map; copying that output literally would suppress background/context values.
That is a poor fit for a study whose premise is to preserve useful object and
surrounding-context evidence, and it conflicts with the preference against
gating. The additive residual is therefore the proposed adaptation, not an
upstream module that can simply be plugged in unchanged.

SAM2 already supplies a predicted probability mask to its memory encoder, so
the added branch may be redundant. A negative result would be informative: it
would argue against extra learned mask-conditioned memory writes under this
training setup. Prior results also show that mask-derived regions can collapse
or amplify prediction errors during autoregressive propagation.

## If Run174 does not pass its frozen continuation rule

1. Freeze Run174's final report before touching the next method.
2. Use the implemented adaptation of MemSAM's published convolutional feature
   path at SAM2's memory-write boundary. Keep the correction additive,
   soft-mask-conditioned, and zero initialized. Recover probabilities using
   SAM2's configured sigmoid scale/bias; keep the original transformed mask
   input unchanged. Keep all original SAM2
   parameters frozen.
3. Compare native SAM2, a parameter-matched mask-neutral residual writer, and
   the soft-mask-conditioned writer. Use identical clips, optimizer updates,
   loss, seed, validation checkpoints, and autoregressive predicted-mask
   writes. Ground-truth masks are loss targets only after the prompt frame.
4. Engineering gate: zero-initialized writer exactly reproduces native outputs
   and stored memory; confirm the mask-probability inversion, all four native attention layers, pointers,
   positions, and memory-write frame IDs are unchanged. Check finite gradients
   only in the added writer and verify a nonzero correction after training.
5. Run the gate on representative real clips, then train and select only on the
   existing train/validation split. Test on seq16–23 only after a written,
   frozen protocol and a passing validation criterion. Treat PolypGen as
   exploratory because these sequences have informed prior method choices.

Primary outcome: sequence-mean present-frame Dice against same-setup native
SAM2 and the mask-neutral writer. Also report per-sequence changes, boundary F,
false alarms on empty frames, re-detection after absence, and runtime. Do not
select masks, thresholds, or a checkpoint to make the result positive.

## Decision

This is the most distinct literature-supported candidate found so far, because
it changes memory *writing* rather than memory reading or token retention. It
has enough rationale for one controlled pilot if Run174 fails, but not enough
evidence to promise a meaningful Dice gain or a publishable contribution.
Publishability would require a clear gain over both native SAM2 and the
capacity-matched mask-neutral writer, stable sequence-level results, and
independent evaluation. If it does not pass those comparisons, stop this branch
of the hypothesis rather than add more regional variants.

## Sources

- [MemSAM paper, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Deng_MemSAM_Taming_Segment_Anything_Model_for_Echocardiography_Video_Segmentation_CVPR_2024_paper.pdf)
- [MemSAM official implementation](https://github.com/dengx10520/MemSAM) (locally inspected at commit `36f1b988f14b518a6346d2edb1267340387e646c`)
- [SAM2 official memory encoder/write path](https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_base.py)
- [SAM2Long, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.pdf), for the broader autoregressive error-accumulation motivation; its tree search is not part of this candidate.
