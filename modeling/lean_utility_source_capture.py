"""Read-only identity and early-feature evidence for raw Lean memory reads.

A source is attributable only if the exact entry from the current output
mapping is materialized and its complete feature and positional sequence has
one unambiguous, non-overlapping match in the attention input. Other keys,
including insurance keys, remain unattributed and cannot be intervened on.
"""
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import torch
from modeling.lean_memory_capture import clone_data
from modeling.lean_memory_position_lookup import MemoryPositionLookup
from modeling.lean_region_read_intervention import locate_verified_span


@contextmanager
def capture_utility_sources(model, target_frames, output, observations):
    feature = model._get_image_feature
    prepare = model._prepare_memory_conditioned_features
    materialize = model._materialize_memory_entry
    attention = model.memory_attention.forward
    state = {'frame': None, 'entries': {}, 'candidates': {}, 'captured': set(), 'features': set()}
    root = Path(output)
    offsets = model.maskmem_tpos_enc.detach().reshape(-1, model.mem_dim).cpu()
    with torch.no_grad():
        grid = model.memory_encoder.position_encoding(torch.zeros(1, model.mem_dim, 64, 64)).flatten(2).permute(2, 0, 1)[:, 0].cpu()
    lookup = MemoryPositionLookup(grid.float(), torch.zeros(1, model.mem_dim))

    def get_feature(self, inference_state, frame_idx, batch_size):
        result = feature(inference_state, frame_idx, batch_size)
        frame = int(frame_idx)
        if frame not in state['features']:
            f = result[1]['backbone_fpn'][0]
            if batch_size != 1 or tuple(f.shape) != (1, 32, 256, 256) or f.device.type != 'cpu' or not torch.isfinite(f).all():
                raise ValueError('Unexpected source feature')
            folder = root / 'source_features'; folder.mkdir(exist_ok=True)
            torch.save({'frame': frame, 'feature': f[0].detach().float().cpu().clone()}, folder / f'{frame:06d}.pt')
            state['features'].add(frame)
        return result

    def read_prepare(self, *args, **kwargs):
        if args or state['frame'] is not None:
            raise ValueError('Unexpected or nested memory preparation')
        frame = int(kwargs['frame_idx']); entries = {}
        for group in ('cond_frame_outputs', 'non_cond_frame_outputs'):
            for index, entry in kwargs['output_dict'][group].items():
                if id(entry) in entries:
                    raise ValueError('Aliased memory entry')
                entries[id(entry)] = (int(index), group)
        state.update(frame=frame, entries=entries, candidates={})
        try:
            return prepare(*args, **kwargs)
        finally:
            state.update(frame=None, entries={}, candidates={})

    def fetch(self, prev, device, *, is_cond=False):
        result = materialize(prev, device, is_cond=is_cond)
        if state['frame'] in target_frames and prev is not None and not is_cond and result[0] is not None:
            identity = state['entries'].get(id(prev))
            if identity is not None and identity[1] == 'non_cond_frame_outputs':
                index = identity[0]
                if index >= state['frame'] or prev.get('frame_idx') != index:
                    raise ValueError('Source identity differs from the resident entry')
                state['candidates'][index] = clone_data({'features': result[0], 'positions': result[1]})
        return result

    def attend(self, *args, **kwargs):
        frame = state['frame']
        if frame in target_frames and frame not in state['captured']:
            if args:
                raise ValueError('Unexpected attention arguments')
            memory = kwargs['memory']; positions = kwargs['memory_pos']; ptr = int(kwargs['num_obj_ptr_tokens'])
            protected = [(0, 4096)]; sources = []; rejected = []
            for index, source in sorted(state['candidates'].items()):
                try:
                    start = locate_verified_span(memory, positions, ptr, source['features'], source['positions'], offsets, protected)
                except ValueError as exc:
                    if str(exc) != 'Source does not have a unique verified read span':
                        raise
                    rejected.append({'frame': index, 'reason': str(exc)})
                    continue
                n = len(source['features']); protected.append((start, start + n))
                ids = lookup.decode(source['positions'].float().cpu(), 0)
                if len(ids) != n or len(torch.unique(ids)) != n:
                    raise ValueError('Source positions are not unique')
                sources.append({'frame': index, 'start': start, 'token_ids': ids, **source})
            folder = root / 'source_snapshots'; folder.mkdir(exist_ok=True)
            torch.save({'frame': frame, 'pointer_count': ptr, 'sources': sources, 'rejected': rejected}, folder / f'{frame:06d}.pt')
            observations.append({'frame': frame, 'attributed_sources': len(sources), 'attributed_tokens': sum(len(s['token_ids']) for s in sources), 'rejected': rejected})
            state['captured'].add(frame)
        return attention(*args, **kwargs)

    model._get_image_feature = MethodType(get_feature, model)
    model._prepare_memory_conditioned_features = MethodType(read_prepare, model)
    model._materialize_memory_entry = MethodType(fetch, model)
    model.memory_attention.forward = MethodType(attend, model.memory_attention)
    try:
        yield
        if state['captured'] != set(target_frames):
            raise ValueError('Source capture coverage incomplete')
        for frame in state['captured']:
            item = torch.load(root / 'source_snapshots' / f'{frame:06d}.pt', map_location='cpu', weights_only=True)
            if any(s['frame'] not in state['features'] for s in item['sources']):
                raise ValueError('Attributed source lacks its early features')
    finally:
        model._get_image_feature = feature
        model._prepare_memory_conditioned_features = prepare
        model._materialize_memory_entry = materialize
        model.memory_attention.forward = attention
