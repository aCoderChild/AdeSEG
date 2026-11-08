"""Training-only SupContrast supervision of Cutie's actual content Q/K maps.

The author loss is imported unchanged. Local glue pools soft GT occupancy into
object/context prototypes within ONE video/object; GT never enters a forward
read. There is no contrastive queue, extra inference module or hard token gate.
"""
import hashlib
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

SUPCON_REVISION = "66a8fe53880d6a1084b2e4e0db0a019024d6d41a"
SUPCON_SHA256 = "a432e24bdff118aa0ab3013a4c85a4b099871ebef554177054d91fe6fca616db"


def author_supcon():
    source = Path(__file__).resolve().parents[2] / "external/SupContrast/losses.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != SUPCON_SHA256:
        raise RuntimeError("SupContrast loss differs from pinned upstream source")
    spec = importlib.util.spec_from_file_location("adeseg_author_supcon", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SupConLoss


def region_prototypes(features, occupancy):
    """Area-weighted features [N,C] for nonempty GT object/context supports.

    An exactly absent class has no prototype, not an invented zero embedding.
    Fractional boundary cells contribute proportionally to both region means.
    """
    if features.ndim != 2 or occupancy.shape != features.shape[:1]:
        raise ValueError("Feature grid and GT occupancy differ")
    if not torch.isfinite(occupancy).all() or (occupancy < 0).any() or (occupancy > 1).any():
        raise ValueError("Expected finite occupancy in [0,1]")
    values, roles = [], []
    for role, weights in ((1, occupancy), (0, 1 - occupancy)):
        area = weights.sum()
        if float(area.detach()) == 0:
            continue
        values.append((features * weights[:, None]).sum(0) / area)
        roles.append(role)
    return values, roles


class ForegroundContextContrast:
    """Observe a four-frame training chunk; no mutation of reader outputs.

    Stored-mask predictions continue to drive actual Cutie pooling. The loss
    separately pools pre-position content features using training GT. It trains
    the SAME Q/K projections used in inference, rather than a discarded head.
    Context is modeled as one class within this single video/object only.
    """
    def __init__(self, reader, observer, targets, weight=.1, temperature=.07):
        if weight < 0 or temperature <= 0:
            raise ValueError("Invalid contrastive objective settings")
        self.reader, self.observer, self.targets = reader, observer, targets
        self.weight = weight
        self.loss_fn = author_supcon()(temperature=temperature, base_temperature=.07)
        self.records, self.handles = [], []
        self.memory = self.query = None

    def _memory(self, module, args, output):
        if self.memory is not None:
            raise RuntimeError("Unconsumed memory features")
        self.memory = output

    def _query(self, module, args):
        if self.query is not None:
            raise RuntimeError("Unconsumed query features")
        self.query = module.norm(args[0])

    def __enter__(self):
        self.handles = [
            self.reader.summarizer.feature_pred.register_forward_hook(self._memory),
            self.reader.reader.register_forward_pre_hook(self._query),
        ]
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.memory = self.query = None

    def __call__(self, sample):
        memory, query = self.memory, self.query
        self.memory = self.query = None
        if memory is None or query is None:
            raise RuntimeError("Missing reader content for contrastive supervision")
        if memory.shape[0] != 1 or query.shape[0] != 1:
            raise ValueError("Train one persistent instance at a time; never mix IDs")
        trace = self.observer.calls[-1]
        frame = sample["frame"]
        ids = [span["frame"] for span in trace["memory_frames"]]
        if trace["frame"] != frame or len(ids) != memory.shape[1] or any(i >= frame for i in ids):
            raise RuntimeError("Contrastive labels do not match causal native memory")
        _, _, h, w, channels = memory.shape
        if query.shape[1] != h * w:
            raise ValueError("Query and memory grids differ")
        mha = self.reader.reader.cross_attn
        weight, bias = mha.in_proj_weight, mha.in_proj_bias
        values, roles = [], []
        pairs = [(query[0], sample["target"], 0)] + [
            (memory[0, j].reshape(-1, channels), self.targets[i], 1)
            for j, i in enumerate(ids)
        ]
        for features, target, projection in pairs:
            occupancy = F.adaptive_avg_pool2d(target.detach().float(), (h, w)).reshape(-1)
            prototypes, labels = region_prototypes(features, occupancy.to(features.device))
            offset = projection * channels
            for prototype, role in zip(prototypes, labels):
                value = F.linear(prototype, weight[offset:offset + channels],
                                 None if bias is None else bias[offset:offset + channels])
                values.append(value)
                roles.append(role)
        # One-role batches contain no foreground/context contrast. Skip only
        # this supervision term; all memory tokens remain in native inference.
        if len(set(roles)) < 2:
            loss = query.sum() * 0
        else:
            features = F.normalize(torch.stack(values).float(), dim=-1, eps=1e-6)
            loss = self.loss_fn(features[:, None], labels=torch.tensor(roles, device=features.device))
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite upstream supervised contrastive loss")
        self.records.append({"frame": frame, "memory_frames": ids,
                             "foreground_prototypes": roles.count(1),
                             "context_prototypes": roles.count(0),
                             "contrastive_loss": float(loss.detach()),
                             "weighted_contrastive_loss": float(loss.detach()) * self.weight,
                             "gt_used_only_after_forward": True})
        return self.weight * loss
