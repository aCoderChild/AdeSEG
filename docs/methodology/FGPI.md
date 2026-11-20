# FGPI — Feature-Guided Prompt Initialization

## Objective

Given a YOLO detection, the module constructs a more informative initial SAM 2 prompt by combining the **detector's original box** with foreground and background points selected from early visual features. The points are not obtained from ground-truth masks.

## Detector-conditioned pseudo-target anchors

Let the detected box be \(b_0=(x_0,y_0,x_1,y_1)\), and let the frozen early image-encoder feature map be \(E_0\in\mathbb R^{C\times H_f\times W_f}\). Each spatial token is normalized to \(z_p=E_0(p)/\|E_0(p)\|_2\).

Define a conservative core \(C(b_0)\) inside the box and an annulus \(N(b_0)\) outside a safety margin. These are *geometric pseudo-regions*, not known polyp and background labels.

A center-weighted interior prototype selects an initial pseudo-anchor. Subsequent anchors encourage feature diversity within a spatially supported candidate pool:

\[
a_{k+1}=\arg\max_{p\in C_s\setminus A_k}
\big[0.7(1-\max_{a\in A_k}\cos(z_p,a))+0.3c(p)\big],
\]

where \(c(p)\) represents box-center support.

The target/background contrast is:

\[
s_{\mathrm{raw}}(p)=\max_{a\in A}\cos(z_p,a)-\cos(z_p,\mu_N),
\quad \mu_N=\operatorname{Normalize}\left(\frac1{|N|}\sum_{q\in N} z_q\right).
\]

The implementation combines \(s_{\mathrm{raw}}\) with 3×3 spatial smoothing, a small center prior for positive points and a near-exterior preference for negative points. Spatial separation discourages clustered prompts.

The prompt is \(\mathcal P_0=\{b_0,P_0^+,P_0^-\}\): one unchanged detector box and fixed numbers of feature-derived positive and negative points. The default uses one of each point type and at most four pseudo-anchors.

**No feature reliability gating:** invalid feature geometry causes an explicit error; contrast strength does not suppress the point prompts or silently substitute a box-only arm.

## Software interfaces

- `adeseg/modules/fgpi/__init__.py`: FGPI / FGPIConfig.
- `adeseg/modules/fgpi/anchors.py`: pseudo-anchor and point scoring.
- `adeseg/modules/fgpi/prompts.py`: SAM 2 prompt construction and native prompt API.
- `scripts/export_yolo_detections.py`: frozen source-hashed YOLO detections.
- `scripts/infer_fgpi_video.py`: automatic MedSAM2 experimental video runner.
- `scripts/evaluate_fgpi_video.py`: initial and post-prompt evaluation.

FGPI's pseudo-anchors are feature candidates **before** the target mask is established; they should not be confused with Lean-SAM2's memory anchors derived from an already initialized object. The scientifically useful test is whether these **specific feature-selected points** outperform equally numerous geometric or random points using the identical detector box.

References: [SAM 2](https://arxiv.org/abs/2408.00714), [Lean-SAM2](https://arxiv.org/abs/2607.19811).
