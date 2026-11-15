"""Region-center memory contrast adapted from the official RegionContrast code."""

import ast
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from modeling.region_tokens import _verified_source

REGIONCONTRAST_REVISION = "7ec8896c738ecb41eeea104976b4649a703a6cd8"
REGIONCONTRAST_SOURCE = "pyseg/models/decoder_contrast.py"
ROLE_COUNT = 3
TOKEN_GRID = (64, 64)
TOKEN_COUNT = TOKEN_GRID[0] * TOKEN_GRID[1]


class _PortableDeviceCall(ast.NodeTransformer):
    """Adapt the author's CUDA-only return to the input tensor's device."""

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "cuda":
            node.func = ast.Attribute(
                value=node.func.value, attr="to", ctx=ast.Load()
            )
            node.args = [
                ast.Attribute(
                    value=ast.Name(id="fea", ctx=ast.Load()),
                    attr="device",
                    ctx=ast.Load(),
                )
            ]
            node.keywords = []
        return node


@lru_cache(maxsize=1)
def author_region_center():
    """Load the repository's region-center constructor with device portability."""
    path, source = _verified_source(
        "RegionContrast", REGIONCONTRAST_REVISION, REGIONCONTRAST_SOURCE
    )
    cls = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef)
        and node.name == "dec_deeplabv3_contrast"
    )
    method = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "construct_region"
    )
    method = _PortableDeviceCall().visit(method)
    ast.fix_missing_locations(method)
    namespace = {"torch": torch}
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["construct_region"]


class RegionContrastBank(nn.Module):
    """Class-wise region-center queue with paper-equation InfoNCE."""

    def __init__(self, channels=256, capacity=2975, temperature=0.2):
        super().__init__()
        if channels != 256 or capacity <= 0 or temperature <= 0:
            raise ValueError("Expected 256 channels and positive queue settings")
        self.channels = channels
        self.capacity = int(capacity)
        self.temperature = float(temperature)
        self.register_buffer("queue", torch.zeros(ROLE_COUNT, capacity, channels))
        self.register_buffer("count", torch.zeros(ROLE_COUNT, dtype=torch.long))
        self.register_buffer("pointer", torch.zeros(ROLE_COUNT, dtype=torch.long))
        self.center = author_region_center()

    @staticmethod
    def _one_hot(labels):
        return F.one_hot(labels.long(), ROLE_COUNT).permute(0, 3, 1, 2).float()

    def _centers(self, features, labels):
        if features.ndim != 4 or features.shape[1:] != (self.channels, *TOKEN_GRID):
            raise ValueError("Expected [batch, 256, 64, 64] projected memory features")
        if labels.shape != (features.shape[0], *TOKEN_GRID):
            raise ValueError("Region labels must align with the token grid")
        centers, classes = self.center(None, features, self._one_hot(labels))
        return {
            int(role): F.normalize(center.float(), dim=0, eps=1e-6)
            for center, role in zip(centers, classes)
        }

    def _queue_values(self, role):
        count = int(self.count[role])
        return self.queue[role, :count].detach()

    def _enqueue(self, centers):
        with torch.no_grad():
            for role, center in centers:
                pointer = int(self.pointer[role])
                self.queue[role, pointer].copy_(center.detach().to(self.queue))
                self.pointer[role] = (pointer + 1) % self.capacity
                self.count[role] = min(int(self.count[role]) + 1, self.capacity)

    def forward(self, query, memory, memory_labels, query_labels, pointer_count=0):
        """Compare each query-role center with matching and mismatching memory roles."""
        if query.shape != (1, TOKEN_COUNT, self.channels):
            raise ValueError("Expected one 64x64 projected query")
        if memory.ndim != 3 or memory.shape[0] <= pointer_count:
            raise ValueError("Expected spatial memory and optional trailing pointers")
        spatial = memory.shape[0] - pointer_count
        if spatial != memory_labels.numel() or spatial % TOKEN_COUNT:
            raise ValueError("Memory labels and complete spatial frames must align")
        frame_count = spatial // TOKEN_COUNT
        if query_labels.shape != TOKEN_GRID:
            raise ValueError("Query labels must be a 64x64 map")

        query_features = query[0].float().transpose(0, 1).reshape(
            1, self.channels, *TOKEN_GRID
        )
        q_centers = self._centers(query_features, query_labels.to(query.device)[None])

        memory_features = memory[:spatial, 0].float().reshape(
            frame_count, TOKEN_COUNT, self.channels
        )
        labels = memory_labels.reshape(frame_count, *TOKEN_GRID).to(memory.device)
        current = []
        for frame_index in range(frame_count):
            frame_features = memory_features[frame_index].transpose(0, 1).reshape(
                1, self.channels, *TOKEN_GRID
            )
            current.append(self._centers(frame_features, labels[frame_index][None]))

        loss_terms = []
        to_enqueue = []
        for centers in current:
            to_enqueue.extend(centers.items())
        live_by_role = {
            role: [centers[role] for centers in current if role in centers]
            for role in range(ROLE_COUNT)
        }
        for role, q_center in q_centers.items():
            positives = live_by_role[role] + list(self._queue_values(role).unbind(0))
            negatives = [
                center
                for other_role in range(ROLE_COUNT)
                if other_role != role
                for center in live_by_role[other_role]
            ]
            negatives.extend(
                center
                for other_role in range(ROLE_COUNT)
                if other_role != role
                for center in self._queue_values(other_role).unbind(0)
            )
            if not positives or not negatives:
                continue
            positive_logits = torch.stack(positives) @ q_center / self.temperature
            negative_logits = torch.stack(negatives) @ q_center / self.temperature
            numerator = torch.logsumexp(positive_logits, dim=0)
            denominator = torch.logsumexp(
                torch.cat((positive_logits, negative_logits)), dim=0
            )
            loss_terms.append(denominator - numerator)

        self._enqueue(to_enqueue)
        if loss_terms:
            loss = torch.stack(loss_terms).mean()
        else:
            loss = query.sum() * 0.0
        return loss, {"query_roles": sorted(q_centers),
                      "memory_role_counts": [len(live_by_role[i]) for i in range(ROLE_COUNT)],
                      "contrast_roles": len(loss_terms)}
