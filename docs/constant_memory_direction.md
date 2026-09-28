# RGM-MedSAM2 design

RGM-MedSAM2 adapts two established ideas to MedSAM2 rather than claiming a new
general video-object-segmentation memory mechanism.

- [RDE-VOS (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Recurrent_Dynamic_Embedding_for_Video_Object_Segmentation_CVPR_2022_paper.pdf)
  recurrently compresses historical memory. Its official
  [implementation](https://github.com/Limingxing00/RDE-VOS-CVPR2022) uses a
  two-frame `Conv3d(..., kernel_size=(2, 3, 3))` memory compressor.
- [LiVOS (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_LiVOS_Light_Video_Object_Segmentation_with_Gated_Linear_Matching_CVPR_2025_paper.pdf)
  uses learned gating in a recurrent memory formulation. Its official
  [implementation](https://github.com/uncbiag/LiVOS) has a different linear
  key-value readout and decoder from MedSAM2.

RGM preserves MedSAM2's memory encoder, memory attention, decoder, and native
object-pointer policy. It replaces only the growing spatial mask-memory list:

```text
candidate C_t = MedSAM2 memory encoder(predicted mask_t)
compressed candidate S_tilde = Conv3d([S_(t-1), C_t])
g_t = sigmoid(MLP([q_t, GAP(S_(t-1)), GAP(C_t), GAP(|C_t - S_(t-1)|)]))
S_t = (1 - g_t) S_(t-1) + g_t S_tilde
```

`q_t` is MedSAM2's decoder-predicted IoU. It is an input to learned reliability
estimation, never a hard frame-selection score and never supervised by
ground-truth IoU. This is necessary because the previous PolypGen diagnostic
showed that predicted IoU is correlated with actual IoU but does not reliably
rank the best individual memory frames.

The initial compressor is candidate identity and the gate is 0.1. This makes
the initial RGM exactly fixed EMA alpha=0.1, providing a controlled baseline.
