"""SAM2's author MLP plus RDE-VOS's unchanged channel-KL loss.

Project glue adapts spatial values before native v_proj, preserving explicitly
identified conditioning frames and pointer tails. No frame/token selection or
confidence gate. This is a SAM2 adaptation, not the full RDE-VOS model.
"""
import ast
from contextlib import contextmanager
import inspect
from pathlib import Path
import subprocess

import torch
from torch import nn
import torch.nn.functional as F

from modeling.region_tokens import SAM2_REVISION, _verified_source
from modeling.native_memory_layout import selected_memory_frames

RDE_REVISION = '548a889395e2370c07962692e01e3abf8f518153'


def _class(source, path, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name)
    namespace = {'nn': nn, 'F': F}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def author_consistency_loss(repository, temperature=1.0):
    """Extract the unchanged, pinned class without its CUDA/DDP trainer imports."""
    repository = Path(repository)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()
    if revision != RDE_REVISION:
        raise RuntimeError('RDE author revision differs')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'model/model.py'],
                   cwd=repository, check=True, capture_output=True)
    path = repository / 'model/model.py'
    return _class(path.read_text(), path, 'KLDivLoss')(temperature=temperature)


class MemoryValueRepair(nn.Module):
    """One shared token MLP, independent for each token and object batch item."""
    def __init__(self, channels=64, hidden=128, grid_shape=(64, 64)):
        super().__init__()
        path, source = _verified_source('sam2', SAM2_REVISION, 'sam2/modeling/sam2_utils.py')
        self.mlp = _class(source, path, 'MLP')(channels, hidden, channels, 2)
        nn.init.zeros_(self.mlp.layers[-1].weight)
        nn.init.zeros_(self.mlp.layers[-1].bias)
        self.channels, self.grid_shape = channels, tuple(grid_shape)
        self._repair_indices = None
        self._spatial_count = None

    def bind_layout(self, spans, conditioning_frames):
        """Bind native frame spans for exactly the next object-aligned read."""
        cursor, indices, seen = 0, [], set()
        conditioning_frames = set(conditioning_frames)
        for span in spans:
            if span['start'] != cursor or span['stop']-cursor != self.grid_shape[0]*self.grid_shape[1] or span['frame'] in seen:
                raise RuntimeError('Malformed or duplicate native frame span')
            seen.add(span['frame'])
            if span['frame'] not in conditioning_frames:
                indices.extend(range(cursor, span['stop']))
            cursor = span['stop']
        if not spans:
            raise RuntimeError('No spatial layout bound')
        self._spatial_count = cursor
        self._repair_indices = tuple(indices)

    def forward(self, memory, pointer_tokens=0):
        if memory.ndim != 3 or memory.shape[-1] != self.channels or not 0 <= pointer_tokens <= memory.shape[0]:
            raise RuntimeError('Invalid sequence-first memory/pointer layout')
        if pointer_tokens == memory.shape[0]:
            return memory
        if self._spatial_count != memory.shape[0]-pointer_tokens or self._repair_indices is None:
            raise RuntimeError('Actual native memory differs from the bound layout')
        if not torch.isfinite(memory).all():
            raise RuntimeError('Nonfinite native value input')
        if not self._repair_indices:
            return memory
        index = torch.tensor(self._repair_indices, dtype=torch.long, device=memory.device)
        raw = memory.index_select(0, index)
        delta = self.mlp(raw.to(self.mlp.layers[0].weight.dtype)).to(memory.dtype)
        if not torch.isfinite(delta).all():
            raise RuntimeError('Nonfinite value residual')
        return memory.index_copy(0, index, raw + delta)


def value_consistency(student, teacher, repair_indices, criterion):
    """RDE channel KL over aligned nonprompt [tokens,object,channels] values.

    Teachers are detached training targets. No labels enter the student module.
    Both inputs keep the identical native frame/token/object order.
    """
    if student.shape != teacher.shape or student.ndim != 3 or not repair_indices:
        raise ValueError('Expected aligned nonempty value tensors and repair indices')
    index = torch.tensor(repair_indices, dtype=torch.long, device=student.device)
    s = student.index_select(0, index).permute(1, 2, 0).float()
    t = teacher.detach().to(student.device).index_select(0, index).permute(1, 2, 0).float()
    if not torch.isfinite(s).all() or not torch.isfinite(t).all():
        raise RuntimeError('Nonfinite consistency input')
    return criterion(s, t)


@contextmanager
def bind_native_reads(model, adapter):
    """Observe each native object bank and bind its actual conditioning spans."""
    original = model._prepare_memory_conditioned_features
    signature = inspect.signature(original)

    def observed(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        if not values['is_init_cond_frame']:
            bank = values['output_dict']
            spans = selected_memory_frames(model, values['frame_idx'], bank, values['num_frames'])
            if spans:
                adapter.bind_layout(spans, bank['cond_frame_outputs'])
            else:
                adapter._spatial_count, adapter._repair_indices = 0, ()
        return original(*args, **kwargs)

    model._prepare_memory_conditioned_features = observed
    try:
        yield
    finally:
        model._prepare_memory_conditioned_features = original
        adapter._spatial_count, adapter._repair_indices = None, None
