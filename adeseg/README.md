# AdeSEG research API

The proposed method has three concise, stable public interfaces:

```python
from adeseg.modules import FGPI, FGPIConfig, RMFA, RMFAConfig, TTMR, TTMRConfig
from adeseg.modules.rmfa import build_rmfa
```

| Research module | API | Inputs and outputs |
|---|---|---|
| **FGPI** | `FGPI().construct(features, box_xyxy, image_hw)` | Frozen early FPN + YOLO box → joint box / positive / negative prompts |
| **RMFA** | `build_rmfa(RMFAConfig())(early, deep, logits)` | Fine features + memory-conditioned deep features + native logits → refined logits |
| **TTMR** | `TTMR(TTMRConfig(budget=B)).select(...)` | Prior memory tokens + anchors + usage + source frames → retained indices |

These research modules reuse existing implementations: FGPI in `adeseg/prompts/`, RMFA in `modeling/structured_feature_aggregation.py` (checkpoint-compatible trainable module), TTMR in `adeseg/memory/target_aware_temporal.py` (offline selection reference).

Supporting code is organized under `backbones/`, `inference/`, `adapters/`, `losses/`, `evaluation/`, and `research/`. No second copy of RMFA's trainable model weights is introduced.

The methodological definition is [here](../docs/methodology/OVERVIEW.md).
