# Video-memory papers: architecture and implementation audit

Date: 2026-11-13. This is a source audit and input-staging check, not an
evaluation of segmentation quality. The project hypothesis concerns useful
subsets within weak past-frame memory, under the native SAM2 memory schedule.

| Method | Video architecture | Relationship to hypothesis | Availability and status |
| --- | --- | --- | --- |
| Puspitasari et al., spatiotemporal pruning | SAM2 video memory attention with spatiotemporal token pruning. | Selective tokens are relevant; efficiency results do not prove utility from poorly segmented frames. | No verified official repository found; not implemented from prose. [Paper](https://pubmed.ncbi.nlm.nih.gov/42599858/). |
| Lean-SAM2 | SAM2 video predictor maintains state across frames; TAMP, TCIM and TARR are used by the author's full system. | Token selection is relevant, but the full system changes memory handling and encoder computation; it is a system baseline, not an isolated test of the hypothesis. | Existing pinned checkout `external/Lean-SAM2` at `8d1cf2618519d291905d22fe859b8302476d7f50`; existing `scripts/run_author_baseline.py` loads the author's full model and bypass. The seq3 engineering smoke below passed; full-cohort comparison is not validated. [Paper](https://arxiv.org/abs/2607.19811), [source](https://github.com/DeawhaleQwQ/Lean-SAM2). |
| Competitive Memory Readout | SAM 3 video tracking with target-versus-competitor memory readout and adaptive restoration. | Query-time discrimination is relevant, but competitor objects and SAM 3 differ from a single-polyp SAM2 setup. | No verified official source or trained release found. No SAM2 transplant is claimed. [Paper](https://arxiv.org/abs/2608.22064). |
| TCMN | Separate one-shot medical-video network: key/value encoders, a temporal memory bank, readout and decoder; first annotated mask initializes later video segmentation. | Temporal memory learning is related, but it is a trained standalone architecture, not a drop-in SAM2 token selector. | Cloned [official source](https://github.com/MedAITech/TCMN) to `external/TCMN`, commit `3463aa4d605cbf0b4eea49d85a6a7964c84867ee`. See limitations below. [Paper](https://arxiv.org/abs/2503.14979). |

## TCMN release limitations

- `eval_generic.py` imports `dataset.generic_test_dataset` and
  `inference_core_yv`, neither of which is in the checkout. Its evaluation also
  assumes CUDA and references local data paths. No trained TCMN `.pth` is in the
  checkout, and no official trained release was identified in this audit.
- `model/losses.py` defines `similar_loss2`, but its assignment to
  `losses['loss_similar']` is commented out. The current `defaultdict(int)`
  contributes zero for that term in this path. Thus this checkout does not
  establish a reproducible implementation of the paper's temporal contrastive
  training. This is an observation about this revision, not a claim about the
  authors' experiments.
- The adapter `scripts/stage_tcmn_polypgen.py` prepares the layout described in
  `eval_generic.py`, starting at the first visible annotation. It exposes only
  that indexed mask to the model, with source-frame mapping saved separately.
  Because the official dataset loader is missing, compatibility with its exact
  parsing behavior remains unverified. It is input preparation, not inference.
- A real seq3 staging check placed 15 ordered JPEG links and one initial indexed
  mask under Drive `outputs/experiment/tcmn_source_review_20261113/seq3_staging`;
  source frame and prompt provenance are recorded there. One synthetic test
  checks prompt selection, ordering, and exclusion of later masks.

## Lean baseline requirements

The existing author runner expects SAM2.1 Hiera B+ and the matching author bypass,
and requires a source-, device- and checkpoint-matched smoke before a full run.
Its signature does not independently hash the Lean configuration and bypass
assets; pinning the clean repository protects the tracked files at launch, but a
stronger persisted asset receipt and full validation are required before relying
on a new comparison. Its frozen source is referenced by historical experiments,
so this audit leaves it unchanged. The previously reported Run170 used selected
Lean components, not the full Lean system, and is not a full-system result.

One-sequence engineering smoke: CPU FP32, seq3, initial mask only, full author
TAMP/TCIM/TARR and bypass. All 14 post-prompt logits were exactly equal on a
fresh-model repeat; parameter integrity and artifact delivery passed. Receipt:
Drive `outputs/experiment/lean_source_review_20261113/seq3_smoke/validation.json`;
the separately recorded asset hashes are in `seq3_asset_receipt.json`. The
14 targets are all polyp-present and their mean Dice is 0.869455; independent
arithmetic over `seq3/frames.jsonl` reproduced this value. There are no empty
targets and no matched native comparison in this smoke.
The runner permits an explicit single-sequence smoke and would accept that receipt
for a full run even though its help describes a two-sequence gate. Do not use this
single-sequence receipt to authorize full inference. This smoke tests execution
and reproducibility, not fidelity against an independent author reference or
comparative efficacy.

Further validation: seq2 completed 59 post-prompt targets with exact repeat
logits and unchanged parameters. Independently running the upstream generic VOS
script on seq3 produced 14/14 identical post-prompt binary masks versus the
project adapter. Source: `tools/vos_inference.py`; its predictor calls the same
`propagate_in_video` entry point, and its alternate `propagate_in_video_wj`
wrapper delegates to that entry point. Receipts are under Drive
`lean_source_review_20261113/seq2_smoke/` and
`lean_source_review_20261113/seq3_official_cli_parity.json`. These checks do
not demonstrate parity with the authors' CUDA runtime or all videos.

Full exploratory comparison started as Drive
`lean_polypgen_full_v1/attempt03/`. Its preflight binds both separate smoke
receipts, seq3 author-CLI parity, author configuration/bypass hashes, checkpoint,
manifest and 21 native B+ reference frame/input files. The wrapper checks
per-sequence image and mask content hashes, frame identities and presence before
computing paired equal-sequence Dice and empty-frame false alarms. Attempt01 was
stopped before inference while adding input-provenance hashes; attempt02 failed
before inference on a transient Drive read cancellation. Both are preserved.
An output directory or preflight alone does not establish completion: require
`full/delivery.json` and `comparison.json` before reporting cohort outcomes.

Completion update (2026-11-14): all 21 eligible sequences and 2,012 paired
targets completed; delivery and comparison receipts exist. An independent
saved-mask audit rescored 4,024 masks and checked frame identity, source mask
hashes, presence, Dice, false alarms and aggregate/bootstrap arithmetic; its
receipt is `independent_mask_audit.json`. The full per-sequence table is
`comparison.md` in attempt03. See the project results log for interpretation.

## Conditions for a meaningful continuation

Obtain the missing official TCMN inference files and trained weights and verify
their provenance; resolve the training-loss discrepancy with author materials.
Locate verified Puspitasari and CMR implementations before adapting either.
For a full Lean comparison, bind all author assets to its validation receipt,
verify author behavior and exact replay, and use matched PolypGen prompts and
frame coverage. Report present-frame Dice and empty-frame false alarms separately;
this previously inspected cohort is exploratory. None of these system-level
comparisons alone proves the within-frame token hypothesis.
