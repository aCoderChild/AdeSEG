"""Check query-dependent readouts on a real, unmodified SAM2 attention call."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling.cfbi_training import replay_training_read
from modeling.region_readout import regional_readouts
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_regional_refinement import load, write


class CaptureLabels(nn.Module):
    def region_context(self, vectors, labels, mode):
        self.labels = labels[None]
        return vectors.new_zeros((1, 192))

    def forward(self, attention, context):
        return attention


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'pilot-dir', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--sample-index', type=int, default=0)
    p.add_argument('--layers', type=int, nargs='+', default=[0, 1, 2, 3])
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    setup = json.loads((args.pilot_dir / 'setup.json').read_text())
    if digest(args.checkpoint) != setup['checkpoint_sha256']:
        raise RuntimeError('Checkpoint mismatch')
    if not args.layers or len(set(args.layers)) != len(args.layers) or any(i not in range(4) for i in args.layers):
        raise ValueError('Layers must be unique indices from 0 to 3')
    row = setup['samples'][args.sample_index]
    sample = load(row['path'], row['sha256'])
    repo, _ = activate_author_repo()
    model = build_author_predictor(SimpleNamespace(method='sam2', device='cpu', checkpoint=args.checkpoint), repo)
    gate = CaptureLabels()
    state = {'active': None, 'reads': []}
    native = F.scaled_dot_product_attention

    def make_before(layer):
        def before(module, args, kwargs):
            if state['active'] is not None:
                raise RuntimeError('Nested cross-attention calls')
            state['active'] = layer
            state['pointers'] = int(kwargs.get('num_k_exclude_rope', 0))
        return before

    def after(module, args, kwargs, output):
        state['active'] = None

    def observed(q, k, v, *args, **kwargs):
        output = native(q, k, v, *args, **kwargs)
        if state['active'] is not None:
            if args or kwargs.get('attn_mask') is not None:
                raise RuntimeError('Unexpected native attention arguments')
            readouts, valid = regional_readouts(q, k, v, gate.labels, state['pointers'], sdpa=native)
            if not torch.isfinite(readouts).all():
                raise RuntimeError('Nonfinite regional readout')
            state['reads'].append({'layer': state['active'], 'pointer_tokens': state['pointers'],
                                   'spatial_tokens': k.shape[-2] - state['pointers'],
                                   'valid_regions': valid.tolist(), 'shape': list(readouts.shape),
                                   'mean_norm_by_region': readouts.norm(dim=-1).mean(dim=(0, 2, 3)).tolist()})
        return output

    hooks = []
    for layer in args.layers:
        attention = model.memory_attention.layers[layer].cross_attn_image
        hooks.append(attention.register_forward_pre_hook(make_before(layer), with_kwargs=True))
        hooks.append(attention.register_forward_hook(after, with_kwargs=True, always_call=True))
    before_hash = fingerprint(sample['output_dict'])['sha256']
    try:
        with torch.inference_mode(), patch.object(F, 'scaled_dot_product_attention', observed):
            logits, trace = replay_training_read(model, gate, sample, 'regional', row['sequence'])
    finally:
        for hook in hooks:
            hook.remove()
    if [read['layer'] for read in state['reads']] != sorted(args.layers) or trace['layers'] != [0, 1, 2, 3]:
        raise RuntimeError('Unexpected native attention call count')
    if not torch.equal(logits, sample['native_logits']) or before_hash != fingerprint(sample['output_dict'])['sha256']:
        raise RuntimeError('Native replay or memory changed')
    write(args.output, {'passed': True, 'sequence': row['sequence'], 'frame': row['frame'],
          'sample_sha256': row['sha256'], 'native_logits_identical': True, 'bank_unchanged': True,
          'read': trace, 'readout': state, 'scope': 'single real read; no modified inference'})


if __name__ == '__main__':
    main()
