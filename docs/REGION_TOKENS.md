# Token utilisation in SAM2 memory

## Research question

Can useful information from individual memory tokens, including parts of a difficult past frame, improve video segmentation without choosing whole frames as good or bad? The desired outcome is a reproducible Dice improvement over native SAM2 on PolypGen, then confirmation on independent data.

## Latest tested method

The training-free candidate adapts the [official Video-ToMe implementation](https://github.com/sjpollard/video-how-do-your-tokens-merge) to SAM2 memory attention. Within each local 4×4 grid tile, it matches projected memory keys from the selected past frames, averages their projected values, and restores the merged values to their original slots before attention. The matched control applies the same operation within each frame.

SAM2's pretrained weights, queries, keys, positions, memory writes, frame schedule, object pointers and decoder remain unchanged. The method retains the original number of attention slots, so this experiment makes no speedup claim. Integration code is an adaptation; the full SAM2 method is not the upstream Video-ToMe model.

## Result and decision

The completed 21-sequence PolypGen evaluation obtained mean present-frame Dice **0.710224** for native SAM2, **0.711331** for the within-frame control and **0.710586** for cross-frame value sharing. The cross-frame gain over native was **+0.000362**, below the predeclared meaningful-improvement threshold and below the control. The frozen screen failed; stop this candidate rather than tune on these sequences.

Seq6 and seq8 informed development, and patient independence is unverified. These results are exploratory and do not establish a clinical or generalisable gain. See [REGION_TOKENS_RESULTS.md](REGION_TOKENS_RESULTS.md) for statistics, validation, the descriptive frame-wise YOLO comparison and limitations.
