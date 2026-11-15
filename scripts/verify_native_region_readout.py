"""Verify an identity-initialized regional residual on a captured SAM2 read."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling.cfbi_training import object_presence_loss
from modeling.native_region_readout import NativeRegionalReadout, RegionalReadoutHeads
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_regional_refinement import load, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'pilot-dir', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--sample-index', type=int, default=0)
    p.add_argument('--mode', choices=('regional', 'shuffled', 'generic'), default='regional')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    setup = json.loads((args.pilot_dir / 'setup.json').read_text())
    if digest(args.checkpoint) != setup['checkpoint_sha256']:
        raise RuntimeError('Checkpoint mismatch')
    row = setup['samples'][args.sample_index]
    sample = load(row['path'], row['sha256'])
    repo, _ = activate_author_repo()
    model = build_author_predictor(SimpleNamespace(method='sam2', device='cpu', checkpoint=args.checkpoint), repo)
    model.requires_grad_(False)
    heads = RegionalReadoutHeads().train()
    before = fingerprint(sample['output_dict'])['sha256']
    with NativeRegionalReadout(model, heads, args.mode, row['sequence']) as adapter:
        output = model.track_step(
            frame_idx=sample['frame'], is_init_cond_frame=False,
            current_vision_feats=sample['vision'], current_vision_pos_embeds=sample['positions'],
            feat_sizes=sample['sizes'], point_inputs=None, mask_inputs=None,
            output_dict=sample['output_dict'], num_frames=sample['num_frames'],
            track_in_reverse=False, run_mem_encoder=False)
    if not torch.equal(output['pred_masks'].detach(), sample['native_logits']):
        raise RuntimeError('Zero residual changed native logits')
    if [x['layer'] for x in adapter.calls] != [0, 1, 2, 3] or before != fingerprint(sample['output_dict'])['sha256']:
        raise RuntimeError('Layer coverage or bank integrity failure')
    loss = object_presence_loss(output['object_score_logits'], sample['target'])
    loss.backward()
    norms = [float(layer.mix.weight.grad.abs().sum()) if layer.mix.weight.grad is not None else 0
             for layer in heads.layers]
    if not all(torch.isfinite(torch.tensor(x)) and x > 0 for x in norms):
        raise RuntimeError(f'Invalid layer gradients: {norms}')
    write(args.output, {'passed': True, 'sequence': row['sequence'], 'frame': row['frame'], 'mode': args.mode,
                       'sample_sha256': row['sha256'], 'native_logits_identical': True,
                       'bank_unchanged': True, 'calls': adapter.calls, 'presence_gradient_l1': norms,
                       'scope': 'single captured read; not a propagation or performance test'})


if __name__ == '__main__':
    main()
