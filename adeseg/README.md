# AdeSEG research package

`adeseg/` is the **single public scientific namespace**. The canonical
three research modules are available from `adeseg.modules`:

```python
from adeseg.modules import FGPI, FGPIConfig, RMFA, RMFAConfig, TTMR, TTMRConfig
from adeseg.modules.rmfa import build_rmfa
```

| Module | Interface | Computational ownership |
|---|---|---|
| **FGPI** | `FGPI().construct(early_features, yolo_box, image_hw)` | Native `adeseg/prompts/` implementation |
| **RMFA** | `build_rmfa(RMFAConfig())(early, deep, native_logits)` | Checkpoint-compatible trainable core in `modeling/` |
| **TTMR** | `TTMR(TTMRConfig(budget=N)).select(...)` | Offline/causal selector in `adeseg/memory/` |

The other subpackages serve the same framework: `backbones/` loads vendor
models, `inference/` integrates hooks, `losses/` and `adapters/` expose
training objectives, and `research/` defines experiment contracts.

**Dependency direction:** new research code imports `adeseg.*`. The
`modeling/` package contains source-bound implementations and is not a
parallel public API. Existing training code may use original `modeling.*`
imports where necessary for checkpoint provenance.

**Canonical locations:** `adeseg/modules/fgpi/{anchors,prompts}.py`,
`adeseg/modules/rmfa.py`, and `adeseg/modules/ttmr/selection.py`.
The older `adeseg/prompts/*` and `adeseg/memory/target_aware_temporal.py`
paths are small compatibility shims, not duplicate algorithms.

The complete architecture is in
[docs/methodology/OVERVIEW.md](../docs/methodology/OVERVIEW.md);
the internal/core boundary is described in [modeling/README.md](../modeling/README.md).
