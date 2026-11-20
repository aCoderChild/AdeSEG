# AdeSEG report-ready methodological framework (frozen at 2026-11-20)

**Purpose:** A defensible pre-experiment architecture and mathematical formulation for the near-term report. Do not interpret this as an executed end-to-end FGPI+RMFA+TTMR result.

## 1. Scientific positioning

The clinical target is segmentation of **adenoid tissue** and the separately annotated **nasopharyngeal airway** in endoscopy video. The currently available public evaluation uses PolypGen, which is *colonoscopy* with a single polyp target: an engineering/domain-shift proxy, not validated adenoid clinical evidence.

The strongest historical reference is **Lean-SAM2 on SAM2.1 Hiera-B+**, under an author implementation with indexed-spatial-RoPE compatibility correction for MPS, one archived *automatically obtained mask* at initialization, and no subsequent redetection. The original saved report has 21/23 promptable sequences, 1,683 GT-positive post-prompt frames, equal-sequence GT-positive Dice **0.7420711278676757**, frame-weighted Dice **0.77249528554744**, and **188 false alarms / 336 GT-empty post-prompt frames**. Seq1/seq7 had no automatic initialization and were excluded. Seq23 used a *retrospective crop*; the cohort had been previously inspected. These limitations must accompany any citation of the result. The archived **0.7421 is neither a FGPI result, nor an RMFA result, nor a TTMR result**, and is not directly comparable with unmatched oracle-mask, YOLO-SAM2, or MedSAM2 experiments.

Primary original data: https://drive.google.com/file/d/1jjqesyv8RXcUI9JN5gcoVVnvSFn9u7O3/view
Repository checkpoint/evidence: experiments/reference_baselines/modified_lean_742.md
Original implementation: backup/experiment-full-20261118

## 2. Research question and error decomposition

Let a sequence be X=(x_0,...,x_T), with binary GT region y_t available only for retrospective supervised learning/evaluation. The deployed predictor receives x_t and causal history, never GT at inference. It outputs per-frame logits z_t and mask \hat y_t = 1[sigmoid(z_t)>0.5].

The proposed research question is: **Can a detector-conditioned prompt, region-aware *residual* segmentation refinement, and target-preserving memory selection address distinct points of failure in causal endoscopy video segmentation, without replacing the pretrained SAM2/Lean-SAM2 decoder?**

Three testable bottlenecks:
1. Initial target ambiguity: the detector's box is imprecise or encloses visually similar background.
2. Spatial confusion: a memory-conditioned deep representation is semantically coherent but local lesion boundaries are blurred or confused.
3. Temporal error accumulation: previous wrong masks and repeated/informationally redundant memory tokens bias subsequent predictions.

These are conceptual failure categories, not a mathematically proven partition of error or three experimentally isolated causes of the 0.742 result. Especially, low per-sequence Dice does not by itself identify the cause.

## 3. One proposed architecture, but two actual code tracks

~~~text
Input frame x_0
  |-- detector D_YOLO(x_0) ----------------------> box b_0
  |-- frozen early image features E_0 ----------> FGPI pseudo-target anchors
  |                                                  |
  +------------------------------------------ box + P+ + P-
                                                     |
                                       SAM2/Lean-SAM2 prompt encoder
                                                     |
Current frame x_t --> frozen image backbone --> memory attention --> native mask logits z_t^0
                          |                       |                 |
                      early FPN e_t       deep memory state h_t    |
                          +-----------------------+-----------------+
                                                  |
                                      RMFA FG/BG channel aggregation
                                                  |
                                       residual delta_t (trained)
                                                  |
                                   z_t = z_t^0 + delta_t
                                                  |
                                segmentation output (output-only first)
                                                  |
                             SAM2/Lean-SAM2 memory encoder/bank
                                                  |
                       native Lean TAMP / TCIM / TARR, plus
                       *proposed* TTMR causal fixed-budget selector
                                                  |
                                        next frame t+1
~~~

**Important code boundary:** The implemented FGPI test runner currently uses MedSAM2 Hiera-T/512; the experimental structured RMFA and memory policies use Lean-SAM2/SAM2.1 Hiera-B+. There is **no validated shared forward pass** executing all three on the archived 0.742 reference backbone. Therefore this diagram is a proposed *unified Lean-SAM2 system*, and any report illustration must label the FGPI-to-Lean adapter as **planned integration**. The historical Lean-SAM2 result is a baseline, not evidence for the dashed/untested paths. Do not silently combine MedSAM2 and Lean results.

## 4. FGPI: detector-conditioned token-based prompt initialization

At the detected first eligible frame t_0, YOLO produces box b=(x_min,y_min,x_max,y_max). An early, frozen visual feature map E in R^{C x H' x W'} is spatially aligned with the source image. Define an interior core C(b) and safely external annulus N(b) using box geometry; C(b) is **not GT foreground**.

For normalized candidate token e_p, construct a *provisional*, center-supported anchor set A={a_k}. A seed token is selected from central core tokens by agreement with the center-weighted core prototype; remaining anchors are greedily diversified within a spatially supported subset. In a simplified form,

  a_(k+1) = argmax_(p in C_s \ A_k) [ 0.7*(1-max_(a in A_k)cos(e_p,a)) + 0.3*c(p) ],

where c(p) denotes normalized box-center support and C_s is the supported core candidate set. This is **inspired by** Lean-SAM2's diverse anchors, but cannot claim known target identity: the initial mask does not exist yet.

Compute foreground anchor similarity s_fg(p)=max_(a in A)cos(e_p,a), background similarity s_bg(p)=cos(e_p,mu_N), and spatially smoothed contrast

  s(p) = 0.65*[s_fg(p)-s_bg(p)] + 0.35*AvgPool_3x3(s_fg-s_bg)(p).

Select fixed positive points P+ inside C(b) and negative points P- outside a safety margin, with spatial separation. Initialize jointly:

  P_0 = { b, P+, P- };  z_0^0 = Decoder( MemoryAttn(E_0,M_0), PromptEncoder(P_0) ).

**No feature-reliability gate:** FGPI always adds its fixed points to a valid detector box, regardless of contrast; impossible geometry raises a recorded eligibility error, not a hidden box-only fallback. The default is one positive and one negative point, with up to four provisional anchors. Detector admission thresholds, if any, must be identical between FGPI and box-only controls. An inaccurate YOLO box can still seed an incorrect pseudo-anchor set.

**Status:** implemented in the separate MedSAM2 detector-conditioned runner, with CPU geometry tests; not integrated/validated with historical Lean B+ or proved better than YOLO box-only, box+geometric points, or box+random points.

## 5. RMFA: region-aware multistage aggregation and residual refinement

RMFA consumes early spatial features e_t, memory-conditioned deep features h_t and the pretrained native logits z_t^0 from the *same* backbone/decoder. 1x1 projections align channels and bilinear interpolation aligns resolution:

  U_t = concat(Up(W_e e_t), Up(W_h h_t)).

Let Q_t=W_Q Pool(U_t), K_t=W_K Pool(U_t), V_t=W_V Pool(U_t), and p_t = Pool(sigmoid(z_t^0)), a predicted—not GT—foreground probability map. The implemented region-conditioned correlations are

  R_fg = Norm(Q_t odot p_t) Norm(K_t odot p_t)^T,
  R_bg = Norm(Q_t odot (1-p_t)) Norm(K_t odot (1-p_t))^T,
  A_t = softmax_channel(R_fg - R_bg).

A_t V_t supplies a learned channel-value correction to the fine feature map. Soft-region prototypes yield spatial contrast C_t = cos(U_t,mu_fg)-cos(U_t,mu_bg), and the residual head predicts a correction

  delta_t = Head(concat(U_t + Up(W_o A_t V_t), C_t, z_t^0)),
  z_t = z_t^0 + delta_t.

The final head is zero-initialized, giving an exact **identity at initialization** when its inputs satisfy the implementation contract. This protects the pretrained output *before training*, but it is **not** a bound on errors after training.

The current offline trainer uses a **boundary-weighted BCE + weighted IoU + soft Dice** objective, with an *optional* GT-positive teacher-preservation term. The latter is a training-only regularizer, not inference-time ground truth. The implementation operates on frozen **TRAIN capture tensors** and does not yet validate a full trained adapter on the 21-video historical cohort.

Crucial design choice: **output-only refinement first**. Do not feed unvalidated refined logits into the memory encoder: otherwise a local mis-segmentation can be amplified by propagation. Future-memory routing is a separate, predeclared ablation, never the default validated method.

**Status:** six-block structured implementation and source-authenticated training/routing infrastructure exist; empirical improvements on a matched Lean dataset are **not** established. The archived 0.742 run did not use RMFA.

## 6. TTMR: target-aware temporal refinement without rebranding Lean's TAMP

Lean-SAM2 already has target-anchored memory pruning (TAMP), temporal condensation with insurance memory (TCIM) and target-aware risk-aware routing (TARR). These are **baseline mechanisms**, not AdeSEG contributions. TTMR is best positioned as a future **conditioned selection rule within the same causal memory budget**, not an independent replacement of all three.

For historical candidate memory tokens m_i from frames t_i<t, let a_0 represent a set of prompt-derived initial target anchors, u_i the normalized native memory-usage score, and r_i a temporal recency term. A candidate score is

  q_i = lambda_a max_(a in A_0) cos(m_i,a)
        + lambda_u u_i + lambda_r exp(-(t-t_i)/tau).

To discourage duplicate tokens, select the next token sequentially using

  i* = argmax_(i not in S) [ q_i - lambda_d max_(j in S) max(0,cos(m_i,m_j)) ]

subject to |S|=B (fixed retained-token budget), no t_i>=t, unique indices, and predeclared per-source-frame lower/upper coverage constraints where feasible.

A fixed B limits the number of selected memory tokens and lets selection quality be compared at matched memory size. **This is a design constraint, not proof of improved segmentation, speed or temporal stability.** Native Lean-TAMP and TCIM already target similar failure modes; the incremental novelty must be tested against them on identical prompts and budgets.

**Status:** CPU/reference offline-shadow selector and multiscale anchor cache experiments exist. No end-to-end evaluated replacement of the live Lean memory policy exists. The historical 0.742 run used the corrected author Lean reference, not this TTMR rule.

## 7. Mathematical design safeguards, not claimed theorems

- **Causality:** all inference prompts, prototypes, and memory selections use the current/past frame only; labels and future frames are excluded.
- **Decoder preservation:** original SAM2/Lean-SAM2 weights remain unchanged; new learned output is a residual. Zero-init produces z_t=z_t^0 at the start of training, not necessarily after learning.
- **Memory budget:** identical B and eligibility across native/TTMR arms prevents mistaking more memory for better selection.
- **Propagation sensitivity:** a heuristic model e_(t+1) <= L e_t + epsilon_t illustrates why poor initialization and noisy memory can accumulate. **L is unknown; this inequality is NOT a proven generalization guarantee for SAM2/Lean-SAM2.** State this as motivation only or omit in a rigorous theorem section.
- **No leakage:** masks may train RMFA on TRAIN and evaluate on DEV/TEST, but cannot choose inference prompts, memory, first eligible detections, or hyperparameters on TEST.

## 8. Evidence and attribution matrix

| Component | Repository implementation | Empirical evidence currently usable | Permissible report wording |
|---|---|---|---|
| Corrected Lean-SAM2 reference | Archived historical author runner | 0.742071 equal-sequence positive Dice; 21/23 eligible; 188/336 GT-empty FP | We observed this exploratory reference result |
| FGPI | YOLO export + MedSAM2 joint box/feature-point runner | CPU geometry/source tests only | We formulate/implement a proposed prompt strategy |
| RMFA | Structured Lean six-block refiner + offline capture trainer | Unit/structural tests; no matched benchmark | We develop a residual multiscale refiner for future validation |
| TTMR | Offline selector + multiscale anchor ablation | CPU toy tests, no deployed measured gain | We investigate a causal memory-selection hypothesis |
| Full FGPI+RMFA+TTMR | **Not implemented as one verified Lean model** | None | We propose a unified architecture; integration remains future work |

### Historical sample difficulty

Original saved report (post-prompt present Dice): seq11=0.2730, seq20=0.1883, seq23=0.0374. Seq23 includes a retrospective crop. These cases motivate failure-mode analysis, **not a statement that any newly proposed module fixes them**.

### Comparisons that must not be written

- Do NOT claim "our FGPI+RMFA+TTMR achieves 0.742".
- Do NOT call 0.742 a trained RMFA or TTMR gain.
- Do NOT directly compare 0.742 with unrelated oracle-GT-initialized or frame-by-frame SAM2 results as an attributed method gain.
- Do NOT claim a clinically validated adenoid obstruction score from PolypGen polyps.
- Do NOT claim target-grounded pseudo-anchors from YOLO alone.

## 9. Minimal follow-up experiment matrix after the deadline

| Arm | Source changes | Primary question | Primary endpoint |
|---|---|---|---|
| A | Original corrected Lean-SAM2, exact same initialization | Reference | Equal-video present Dice + empty FP |
| B | Lean + matched YOLO box only | New automatic reference | Initial prompt-frame Dice |
| C | Lean + box + two geometry points (same budget) | Do points alone help? | Initial prompt-frame paired Delta |
| D | Lean + FGPI target-anchor points (same budget) | Does feature ranking help? | Initial prompt-frame paired Delta |
| E | D + trained RMFA, output-only | Does learned residual help? | Post-prompt Dice + boundary F |
| F | E + RMFA memory routing | Does correction amplify drift? | Long-horizon Dice + FP |
| G | E + TTMR at matched B | Does selection add to native TAMP/TCIM? | Long-horizon Dice + memory cost |

The archived automatic *mask* initialization used for the 0.742 run is not necessarily matched to B-G's YOLO-box policy; any comparison must be re-executed on equal conditions. Patient/source independence, external held-out validation, and confidence intervals over independent units are required for causal effectiveness claims.

## 10. Literature anchoring / overlap

- Ravi et al., SAM 2: Segment Anything in Images and Videos (ICLR 2025): streaming memory, prompt encoder and pretrained mask decoder. https://arxiv.org/abs/2408.00714
- Ouyang et al., Lean-SAM2: Target-Anchored Memory and Encoder Acceleration for SAM2 (2026): native TAMP / TCIM / TARR. https://arxiv.org/abs/2607.19811
- Ding et al., SAM2Long: Enhancing SAM 2 for Long Video Segmentation with a Training-Free Memory Tree (ICCV 2025): uncertainty-aware long-term selection; temporal improvements already studied. https://openaccess.thecvf.com/content/ICCV2025/html/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.html
- CMSA-Net: Causal Multi-scale Aggregation with Adaptive Multi-source Reference for Video Polyp Segmentation (MICCAI 2026): competing multi-scale aggregation and dynamic reference design. https://papers.miccai.org/miccai-2026/0176-Paper1232.html
- PolypGen (Ali et al., 2021): source proxy dataset, cannot validate nasopharyngeal anatomy. Cite existing ali2021polypgen bibliography key.

**Originality boundary:** combined box+points, foreground/background feature correlation, and memory-anchor selection all have predecessors. AdeSEG's possible novelty is their **precisely defined medical-video adaptation and demonstrated value at matched resource/prompt budgets**, neither of which can yet be claimed as an empirical result.

## 11. Report delivery checklist (current week)

1. Insert the architecture diagram with explicit labels "reference measured", "implemented / not benchmarked" and "proposed / offline only".
2. Present the 0.742 reference table **and** the false-alarm/eligibility caveats, with a link to the original report.
3. Include the FGPI, RMFA, TTMR mathematical definitions above, not supposed measured gains.
4. Provide implementation-to-proposal traceability and a transparent limitations/next-experiments section.
5. Freeze claims and avoid new checkpoint-affecting architecture changes until after submission.
