# Efficient-SAM2 device adaptation

Upstream: `jingjing0419/Efficient-SAM2`, commit
`8e1ad06089b9c958ff5ebfc72b9b7d0b904f5ce5`.

Only `SAM2VideoPredictor.init_memory_info()` changes: the uniform-mask buffer
uses `device=self.device` instead of hard-coded CUDA. Shape, dtype and values
are unchanged. The SMR-only adapter disables frame pruning and image routing;
their other hard-coded CUDA branches are not exercised.

The adapter verifies this exact one-line difference against the pinned commit.
Native author retrieval, saliency, top-k selection, FIFO masks, projections,
RoPE and attention remain unchanged. MPS is not considered validated until the
real replay, dense-encoder and active-pruning checks pass.

After initializing the pinned submodules, reproduce the allocation edit from
the AdeSEG repository root:

```sh
git -C external/Efficient-SAM2 apply --check ../../patches/efficient-sam2-device.patch
git -C external/Efficient-SAM2 apply ../../patches/efficient-sam2-device.patch
```

Skip applying if already patched. The submodule remains at its author commit;
the parent repository tracks the patch rather than an unpublished vendor commit.
