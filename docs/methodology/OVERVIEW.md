# Proposed framework: automatic video polyp segmentation

## 1. Problem formulation

Let \(\mathcal X=\{X_t\}_{t=0}^{T}\) be a colonoscopy video and \(Y_t\in\{0,1\}^{H\times W}\) the reference polyp mask. During inference the system receives current/past video frames, detector predictions, image features and causal temporal memory; it does not receive ground-truth masks.

A YOLO detector returns an initial box \(b_0\). The pretrained SAM 2 image encoder yields early spatial features \(E_t\), and Lean-SAM2 supplies deep memory-conditioned representations \(H_t\), native mask logits \(Z_t^{(0)}\), and a memory state \(M_t\).

The proposed method decomposes into:

\[
\begin{aligned}
\mathcal P_0 &= \mathrm{FGPI}(b_0,E_{t_0}),\\
(H_t,Z_t^{(0)}) &= \mathrm{LeanSAM2}(X_t,\mathcal P_0,M_{t-1}),\\
Z_t &= Z_t^{(0)}+\Delta Z_t,\quad \Delta Z_t=\mathrm{RMFA}(E_t,H_t,Z_t^{(0)}),\\
M_t &= \mathrm{TTMR}(M_{t-1},H_t,\mathcal A_0;B),\\
\hat Y_t &= \mathbb 1[\sigma(Z_t)>0.5].
\end{aligned}
\]

Here \(\mathcal A_0\) denotes detector-derived *provisional* target anchors, and \(B\) is the temporal token budget. This defines the unified **research architecture**; the respective implementation paths are detailed below.

## 2. Scientific decomposition

| Stage | Question | Research mechanism |
|---|---|---|
| [FGPI](FGPI.md) | How should an automatic object prompt be initialized? | Box-conditioned, feature-diverse pseudo-anchors and positive/negative points |
| [RMFA](RMFA.md) | How can spatial and memory-conditioned features refine the mask? | FG/BG-aware channel attention, multistage fusion, residual logits |
| [TTMR](TTMR.md) | Which past representations should occupy finite memory? | Causal target relevance, token diversity, frame coverage |

The three methods address distinct variables: **prompt initialization**, **spatial prediction**, and **temporal memory selection**. This allows separate controlled ablations.

## 3. Inference and training strategy

FGPI extracts frozen image features and selects prompts with a fixed positive/negative point budget. The original detector box and derived points are jointly supplied to the native prompt encoder; the box-only arm is a separate control, not an automatic fallback.

RMFA is the learnable branch. It takes the native prediction as its region prior and returns a residual correction, with a zero-initialized final layer. Training uses segmentation and boundary-sensitive losses on correctly partitioned training captures. Initially, corrected logits should affect the **output only**, to isolate boundary refinement from feedback-induced propagation drift.

TTMR starts from a parameter-free, causal reference selection rule and compares retained memory at the same token budget as native Lean-SAM2. Live memory integration must maintain the native key/value, positional and frame-index conventions.

## 4. Tensor and software contracts

| Tensor | Shape | Interpretation |
|---|---|---|
| Early features | \([B,C_e,H_e,W_e]\) | Spatial FPN detail |
| Deep features | \([B,C_d,H_d,W_d]\) | Memory-conditioned target context |
| Native/refined logits | \([B,1,H,W]\) | Single-target mask |
| Memory tokens | \([N,C_m]\) | Historical flattened token vectors |
| Target anchors | \([K,C_m]\) | Memory-space target reference |
| Retained indices | \([B_m]\) | Fixed-budget, causal memory selection |

FGPI uses *image-feature-space* anchors, whereas TTMR uses *memory-token-space* anchors; a shared target identity requires an explicit projection or re-extraction in the appropriate space, not an assumed equality of channel dimensions.

## 5. Integration interfaces

\`adeseg/modules/\` exposes names aligned with the scientific contributions. The FGPI video runner currently uses MedSAM2 Hiera-T; the RMFA and Lean memory paths use SAM2.1 Hiera-B+. A shared end-to-end Lean-SAM2 implementation requires adapting FGPI feature extraction and confirming the memory-routing hook. These are interface requirements, not changes to the mathematical proposal.

See [contributions](CONTRIBUTIONS.md) and [evaluation protocol](../experiments/PLAN.md).
