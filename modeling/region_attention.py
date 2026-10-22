"""Read-only regional measurement using Efficient-SAM2's author Attention.forward.

No replacement attention is used for model predictions. The author function
recomputes weights from native post-projection, post-RoPE Q/K in query chunks.
Only aggregation and observer wiring are project-specific integration code.
"""

import ast
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch

from modeling.region_tokens import REGION_NAMES

EFFICIENT_SAM2_REVISION = "8e1ad06089b9c958ff5ebfc72b9b7d0b904f5ce5"


def author_attention_forward():
    """Load the unchanged author class without importing its CUDA model stack."""
    repo = Path(__file__).resolve().parents[1] / "external/Efficient-SAM2"
    source = "sam2/modeling/sam/transformer.py"
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if revision != EFFICIENT_SAM2_REVISION:
        raise ValueError("Unreviewed Efficient-SAM2 revision")
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", source],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    tree = ast.parse((repo / source).read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Attention"
    )
    namespace = {"torch": torch, "nn": torch.nn, "Tensor": torch.Tensor, "math": math}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(repo / source), "exec"),
        namespace,
    )
    return namespace["Attention"].forward


def attention_summary(key_mass, labels):
    """Report native mass and spatial enrichment; pointers remain separate.

    Enrichment = (region mass / spatial mass) / (region keys / spatial keys).
    Empty regions or zero spatial mass have undefined (None) enrichment.
    """
    mass = torch.as_tensor(key_mass).double().cpu()
    labels = torch.as_tensor(labels).long().cpu()
    if mass.ndim != 1 or mass.shape != labels.shape:
        raise ValueError("Key masses and region labels must align")
    if not torch.isfinite(mass).all() or (mass < 0).any():
        raise ValueError("Invalid attention mass")
    if not ((labels >= -1) & (labels <= 2)).all():
        raise ValueError("Unknown region or pointer label")
    if abs(float(mass.sum()) - 1) > 1e-4:
        raise ValueError("Attention mass does not sum to one")
    spatial = labels >= 0
    count = int(spatial.sum())
    spatial_mass = float(mass[spatial].sum())
    regions = {}
    for index, name in enumerate(REGION_NAMES):
        selected = labels == index
        n = int(selected.sum())
        value = float(mass[selected].sum())
        fraction = n / count if count else None
        share = value / spatial_mass if spatial_mass else None
        regions[name] = {
            "tokens": n,
            "mass": value,
            "mass_per_token": value / n if n else None,
            "spatial_token_fraction": fraction,
            "spatial_attention_share": share,
            "enrichment": share / fraction if n and spatial_mass else None,
        }
    return {
        "regions": regions,
        "spatial_mass": spatial_mass,
        "pointer_mass": float(mass[~spatial].sum()),
        "pointer_tokens": int((~spatial).sum()),
    }


class RegionAttentionObserver:
    """Temporarily observe cross-attention; return native output unchanged."""

    def __init__(self, memory_attention, label_loader, chunk_size=128):
        self.module = memory_attention
        self.label_loader = label_loader
        self.chunk_size = chunk_size
        self.author = author_attention_forward()

        # Q/K/V are already projected and heads already separated by native SAM2.
        def identity(value):
            return value

        self.adapter = SimpleNamespace(
            q_proj=identity,
            k_proj=identity,
            v_proj=identity,
            out_proj=identity,
            _separate_heads=lambda value, heads: value,
            _recombine_heads=identity,
            num_heads=1,
            training=False,
            dropout_p=0.0,
        )
        self.rows = []
        self.label_schemes = {}
        self.enabled = True

    def _context(self, module, args, kwargs):
        units = kwargs.get("memory_units")
        if not units:
            raise ValueError("Missing native memory-key layout")
        pointers = int(kwargs.get("num_obj_ptr_tokens", 0))
        spatial_end = 0
        parts = {}
        for unit in units:
            if unit["start"] != spatial_end or unit["end"] - unit["start"] != 4096:
                raise ValueError("Unexpected native spatial-memory layout")
            loaded = self.label_loader(unit["frame_idx"])
            schemes = loaded if isinstance(loaded, dict) else {"predicted_0.02": loaded}
            if parts and set(parts) != set(schemes):
                raise ValueError("Inconsistent label schemes across memory frames")
            for name, label in schemes.items():
                label = label.flatten()
                if label.numel() != 4096:
                    raise ValueError("Memory-region labels do not match spatial keys")
                parts.setdefault(name, []).append(label)
            spatial_end = unit["end"]
        self.label_schemes = {
            name: torch.cat(values + [torch.full((pointers,), -1)])
            for name, values in parts.items()
        }
        self.labels = self.label_schemes["predicted_0.02"]
        self.units = [
            {k: unit[k] for k in ("start", "end", "frame_idx", "conditioning")}
            for unit in units
        ]

    def _observe(self, original, layer, q, k, v, attn_mask=None):
        output = original(q, k, v, attn_mask=attn_mask)
        if not self.enabled:
            return output
        if attn_mask is not None or self.module.training:
            raise ValueError(
                "Read-only measurement requires unmodified evaluation attention"
            )
        if self.labels.numel() != k.shape[-2] or q.shape[0] != 1:
            raise ValueError("Attention keys and saved spatial labels disagree")
        mass = torch.zeros(k.shape[-2], dtype=torch.float64)
        max_error = 0.0
        for start in range(0, q.shape[-2], self.chunk_size):
            stop = start + self.chunk_size
            reconstructed, (_, _, weights) = self.author(
                self.adapter, q[..., start:stop, :], k, v
            )
            expected = output[..., start:stop, :]
            error = float((reconstructed - expected).abs().max().cpu())
            max_error = max(max_error, error)
            if not torch.allclose(reconstructed, expected, atol=1e-4, rtol=1e-3):
                raise RuntimeError(f"Author attention reconstruction failed: {error}")
            mass += weights.float().sum((0, 1, 2)).cpu().double()
        mass /= q.shape[0] * q.shape[1] * q.shape[-2]
        self.rows.append(
            {
                "layer": layer,
                "query_count": q.shape[-2],
                "heads": q.shape[1],
                "key_count": k.shape[-2],
                "reconstruction_max_abs": max_error,
                "memory_units": self.units,
                **attention_summary(mass, self.labels),
                **(
                    {
                        "label_schemes": {
                            name: attention_summary(mass, labels)
                            for name, labels in self.label_schemes.items()
                        }
                    }
                    if len(self.label_schemes) > 1
                    else {}
                ),
            }
        )
        return output

    def __enter__(self):
        if self.module.utility_controller is not None or self.module.capture_attention:
            raise ValueError("Disable interventions and legacy full-matrix capture")
        self.handle = self.module.register_forward_pre_hook(
            self._context, with_kwargs=True
        )
        self.originals = []
        for index, layer in enumerate(self.module.layers):
            attention = layer.cross_attn_image
            original = attention._attention
            self.originals.append((attention, original))

            def observe(q, k, v, attn_mask=None, original=original, index=index):
                return self._observe(original, index, q, k, v, attn_mask)

            attention._attention = observe
        return self

    def __exit__(self, *exc):
        self.handle.remove()
        for attention, original in self.originals:
            attention._attention = original
