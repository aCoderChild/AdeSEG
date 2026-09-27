# Constant-memory research direction

## Evidence base

- [LiVOS (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_LiVOS_Light_Video_Object_Segmentation_with_Gated_Linear_Matching_CVPR_2025_paper.pdf)
  reformulates memory matching as a gated linear recurrent state. Its
  [official implementation](https://github.com/uncbiag/LiVOS) computes keys,
  values, a gate, readout, and mask decoding in one jointly trained model.
- [RDE-VOS (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Recurrent_Dynamic_Embedding_for_Video_Object_Segmentation_CVPR_2022_paper.pdf)
  maintains a constant-size recurrent embedding with a learned aggregation
  module and trains it for long-video robustness. Its
  [official repository](https://github.com/Limingxing00/RDE-VOS-CVPR2022)
  includes self-correction and guidance components beyond recurrent averaging.
- [XMem (ECCV 2022)](https://computer-vision-in-the-wild.github.io/eccv-2022/static/eccv2022/camera_ready/xmem_cvinw_eccv22.pdf)
  uses multiple connected memory stores: sensory, working, and compact
  long-term memory. Its [official repository](https://github.com/hkchengrex/XMem)
  is a useful reference for bounded multi-store memory.
- [PNS+](https://arxiv.org/abs/2203.14291) uses a global encoder for an anchor
  frame and a local encoder for successive frames in video polyp segmentation.

## What the LiVOS reference actually computes

For normalized keys ``K`` and values ``V``, the public LiVOS code maintains a
state ``S = sum(K^T V)`` and a spatial key sum. It reads a query before writing
the current frame:

```text
R = (Q S) / <Q, key_sum>
S <- diag(g) S + K^T V
key_sum <- key_sum + K
```

Keys use a softmax across the key-channel dimension. The gate is produced by a
learned 1x1 convolution on coarse image features, followed by sigmoid and a
spatial mean. This differs from a raw spatial EMA. The reusable mathematical
state is implemented in `modeling/gated_linear_memory.py`.

## MedSAM2 integration boundary

MedSAM2 memory attention accepts a list of past spatial
`maskmem_features` plus positional encodings. It has no input for a LiVOS
linear-state readout. LiVOS, in contrast, sends its readout directly into its
own jointly trained pixel fusion and mask decoder.

Therefore the following is **not implemented** and must not be claimed:

```text
LiVOS state -> MedSAM2 memory attention
```

Doing so requires a trainable, experimentally validated bridge from MedSAM2
query features and memory-encoder values to a LiVOS readout, followed by a
decoder-conditioning design. That bridge is a new method, not a direct use of
the LiVOS mechanism.

## Safe next experiment

Before designing that bridge, evaluate a bounded native-memory baseline that
retains an anchor frame and a recent reliable frame using MedSAM2's normal
memory interface. Compare it with native MedSAM2 under identical prompts,
then decide whether a recurrent key-value component is justified.
