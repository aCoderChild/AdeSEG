"""Explore spatial-memory and pointer swaps on verified frame-66 histories."""
import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from evaluation.davis2017.metrics import db_eval_boundary
from evaluation.segmentation import calculate_scores
from modeling.cfbi_region_gate import NativeCFBIGate, RegionGate
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_regional_refinement import load, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'native-dir', 'regional-dir', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    folders = {'native': args.native_dir, 'regional': args.regional_dir}
    samples, references = {}, {}
    for name, folder in folders.items():
        if not json.loads((folder / 'completion.json').read_text())['complete']:
            raise RuntimeError(f'{name} snapshot has not passed its replay checks')
        references[name] = json.loads((folder / 'results.json').read_text())
        if digest(args.checkpoint) != references[name]['checkpoint_sha256']:
            raise RuntimeError('Checkpoint mismatch')
        samples[name] = load(folder / 'sample.pt')
    a, b = samples['native'], samples['regional']
    if a['frame'] != 66 or b['frame'] != 66 or a['num_frames'] != b['num_frames']:
        raise RuntimeError('Frame mismatch')
    for key in ('vision', 'positions', 'sizes'):
        if fingerprint(a[key])['sha256'] != fingerprint(b[key])['sha256']:
            raise RuntimeError(f'Target image feature mismatch: {key}')
    for bucket in a['output_dict']:
        if set(a['output_dict'][bucket]) != set(b['output_dict'][bucket]):
            raise RuntimeError(f'History frame IDs differ: {bucket}')
    original = {k: fingerprint(v)['sha256'] for k, v in samples.items()}
    repo, _ = activate_author_repo()
    model = build_author_predictor(SimpleNamespace(method='sam2', device='cpu', checkpoint=args.checkpoint), repo)
    frames = {**load_manifest(ROOT/'data/polypgen_sequence.jsonl', 'dev'), **load_manifest(ROOT/'data/polypgen_sequence.jsonl', 'test')}['seq13']
    gt = load_label_mask(frames[66].mask_path, [1]) == 1
    results = {}
    with torch.inference_mode():
        for spatial in ('native', 'regional'):
            for pointers in ('native', 'regional'):
                sample = copy.deepcopy(samples[spatial])
                for bucket, frameset in sample['output_dict'].items():
                    for index, frame in frameset.items():
                        frame['obj_ptr'] = samples[pointers]['output_dict'][bucket][index]['obj_ptr'].clone()
                with NativeCFBIGate(model, RegionGate().eval(), 'regional', 'seq13') as observer:
                    output = model.track_step(frame_idx=66, is_init_cond_frame=False,
                        current_vision_feats=sample['vision'], current_vision_pos_embeds=sample['positions'],
                        feat_sizes=sample['sizes'], point_inputs=None, mask_inputs=None,
                        output_dict=sample['output_dict'], num_frames=sample['num_frames'],
                        track_in_reverse=False, run_mem_encoder=False)
                if len(observer.trace) != 1 or observer.trace[0]['layers'] != [0, 1, 2, 3]:
                    raise RuntimeError('Incomplete attention trace')
                logits, trace = output['pred_masks'], observer.trace[0]
                key = f'{spatial}_spatial_{pointers}_pointers'
                pred = F.interpolate(logits.float(), size=gt.shape, mode='bilinear', align_corners=False)[0, 0].cpu().numpy() > 0
                results[key] = {'low_res_sha256': tensor_hash(logits), 'dice': float(calculate_scores(pred, gt)['dice']),
                                'boundary_f': float(db_eval_boundary(pred, gt)),
                                'object_score': float(output['object_score_logits'].item()), 'read': trace}
                if spatial == pointers and results[key]['low_res_sha256'] != references[spatial]['arms']['zero']['low_res_sha256']:
                    raise RuntimeError(f'Unmixed replay differs: {key}')
                if spatial == pointers and results[key]['object_score'] != references[spatial]['arms']['zero']['object_score']:
                    raise RuntimeError(f'Unmixed object score differs: {key}')
    if any(fingerprint(v)['sha256'] != original[k] for k, v in samples.items()):
        raise RuntimeError('Source snapshots changed')
    if len({(tuple(v['read']['memory_frames']), v['read']['spatial_keys'], v['read']['pointer_keys']) for v in results.values()}) != 1:
        raise RuntimeError('Attention layouts differ')
    write(out/'results.json', {'exploratory': True, 'sequence': 'seq13', 'frame': 66, 'conditions': results})
    write(out/'completion.json', {'complete': True, 'unmixed_replays_exact': True, 'source_snapshots_unchanged': True})

if __name__ == '__main__':
    main()
