"""Verify native two-frame memory feedback and readout gradients on saved reads."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.cfbi_training import clone_tensors
from modeling.decoder_training_mask import decoder_training_mask, present_mask_loss, select_decoder_mask
from modeling.native_region_readout import NativeRegionalReadout, RegionalReadoutHeads
from modeling.regional_refinement import refinement_loss
from datasets.io import load_label_mask
from datasets.video import load_manifest
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_regional_refinement import write


def step(model, sample, bank, encode):
    return model.track_step(
        frame_idx=sample['frame'], is_init_cond_frame=False,
        current_vision_feats=sample['vision'], current_vision_pos_embeds=sample['positions'],
        feat_sizes=sample['sizes'], point_inputs=None, mask_inputs=None,
        output_dict=bank, num_frames=sample['num_frames'], track_in_reverse=False,
        run_mem_encoder=encode)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'first-dir', 'second-dir', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--loss', choices=('score', 'mask'), default='score')
    a = p.parse_args()
    out = validate_output(a.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    first = torch.load(a.first_dir / 'sample.pt', map_location='cpu', weights_only=False)
    second = torch.load(a.second_dir / 'sample.pt', map_location='cpu', weights_only=False)
    r1 = json.loads((a.first_dir / 'results.json').read_text())
    r2 = json.loads((a.second_dir / 'results.json').read_text())
    if first['frame'] + 1 != second['frame'] or r1['sequence'] != r2['sequence']:
        raise ValueError('Expected consecutive reads from one sequence')
    if any(digest(a.checkpoint) != r['checkpoint_sha256'] for r in (r1, r2)):
        raise ValueError('Checkpoint mismatch')
    repo, revision = activate_author_repo()
    model = build_author_predictor(SimpleNamespace(method='sam2', device='cpu', checkpoint=a.checkpoint), repo)
    model.requires_grad_(False)
    heads = RegionalReadoutHeads()
    bank = clone_tensors(first['output_dict'])
    with NativeRegionalReadout(model, heads, 'regional', r1['sequence']) as observer:
        first_out = step(model, first, bank, True)
        memory = first_out['maskmem_features'].to(torch.bfloat16)
        memory.retain_grad()
        stored = {k: first_out[k] for k in ('maskmem_pos_enc', 'pred_masks', 'obj_ptr', 'object_score_logits')}
        stored['maskmem_features'] = memory
        native = second['output_dict']['non_cond_frame_outputs'][first['frame']]
        for key in ('maskmem_features', 'pred_masks', 'obj_ptr', 'object_score_logits'):
            if not torch.equal(stored[key].detach(), native[key]):
                raise RuntimeError(f'First-frame {key} differs from native')
        bank['non_cond_frame_outputs'][first['frame']] = stored
        with decoder_training_mask(model.sam_mask_decoder) as masks:
            second_out = step(model, second, bank, False)
    if tensor_hash(second_out['pred_masks']) != r2['arms']['zero']['low_res_sha256']:
        raise RuntimeError('Second-frame zero readout differs from native')
    if [x['layer'] for x in observer.calls] != [0, 1, 2, 3] * 2:
        raise RuntimeError('Missing memory attention layers')
    if a.loss == 'mask':
        manifest = ROOT / 'data/polypgen_sequence.jsonl'
        videos = {**load_manifest(manifest, 'dev'), **load_manifest(manifest, 'test')}
        target = torch.from_numpy((load_label_mask(videos[r1['sequence']][second['frame']].mask_path, [1]) == 1).copy())[None, None].float()
        logits = select_decoder_mask(masks)
        target = F.interpolate(target, size=logits.shape[-2:], mode='nearest')
        if not target.any():
            raise RuntimeError('Mask gradient gate requires a present target')
        loss = present_mask_loss(logits, target, refinement_loss)
    else:
        loss = second_out['object_score_logits'].sum()
    loss.backward()
    head_grad = [float(x.mix.weight.grad.abs().sum()) for x in heads.layers]
    memory_grad = float(memory.grad.abs().float().sum()) if memory.grad is not None else 0.0
    if not all(torch.isfinite(torch.tensor(x)) and x > 0 for x in head_grad + [memory_grad]):
        raise RuntimeError('Missing or nonfinite feedback gradient')
    write(out / 'completion.json', {'passed': True, 'sequence': r1['sequence'],
          'frames': [first['frame'], second['frame']], 'sam2_revision': revision,
          'checkpoint_sha256': digest(a.checkpoint), 'first_memory_and_prediction_identical': True,
          'second_prediction_identical': True, 'layers_per_frame': 4,
          'head_gradient_l1': head_grad, 'stored_memory_gradient_l1': memory_grad,
          'loss_type': a.loss, 'loss_value': float(loss.detach()),
          'scope': 'two-frame engineering gate with full memory gradient; not a training or accuracy result'})


if __name__ == '__main__':
    main()
