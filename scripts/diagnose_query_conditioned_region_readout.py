"""Replay the trained query-region gate on fixed native memory histories."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from evaluation.segmentation import calculate_scores
from modeling.native_region_readout import NativeRegionalReadout
from modeling.query_conditioned_region_readout import QueryConditionedRegionHeads
from modeling.cfbi_training import capture_training_read
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, prompt_index, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import load, write

TARGETS = {'seq17': 17, 'seq18': 5, 'seq21': 14}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--adapter-gate', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.run_dir = Path(args.run_dir).resolve()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        run = json.loads((args.run_dir / 'setup.json').read_text())
        completed = json.loads((args.run_dir / 'completion.json').read_text())
        if not completed.get('complete'):
            raise RuntimeError('Source pilot is not complete')
        for relative, expected in run['sources'].items():
            if digest(ROOT / relative) != expected:
                raise RuntimeError(f'Frozen source mismatch: {relative}')
        if digest(args.checkpoint) != run['checkpoint_sha256']:
            raise RuntimeError('Checkpoint mismatch')
        if digest(args.adapter_gate) != run['adapter_gate_sha256']:
            raise RuntimeError('Query-region gate receipt mismatch')
        if not json.loads(args.adapter_gate.read_text()).get('passed'):
            raise RuntimeError('Query-region engineering gate did not pass')
        gate = json.loads(args.gate.read_text())
        if not gate.get('passed') or gate.get('checkpoint_sha256') != run['checkpoint_sha256']:
            raise RuntimeError('Optimizer gate mismatch')

        manifest = ROOT / 'data/polypgen_sequence.jsonl'
        if digest(manifest) != run['manifest_sha256']:
            raise RuntimeError('Manifest mismatch')
        videos = {**load_manifest(manifest, 'dev'), **load_manifest(manifest, 'test')}
        repo, revision = activate_author_repo()
        if revision != run['sam2_revision']:
            raise RuntimeError('SAM2 repository revision changed')
        args.method, args.device = 'sam2', 'cpu'
        args.checkpoint = args.checkpoint.resolve()
        args.model_config = run['protocol'].get('model_config', run['protocol'].get('model', {}).get(
            'configuration', 'configs/sam2.1/sam2.1_hiera_b+.yaml'))
        args.output_dir = args.output_dir
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        seed = run['protocol']['data']['seed']
        reports = {}
        write(args.output_dir / 'setup.json', {
            'source_run': str(args.run_dir), 'source_checkpoint_sha256': run['checkpoint_sha256'],
            'sam2_revision': revision,
            'targets': TARGETS,
            'selection': 'First present-GT frame with Dice 0 in the completed regional rollout for seq17, seq18 and seq21.',
            'scope': 'Outcome-informed fixed-history diagnosis; not an independent performance estimate.'})

        for sequence, target_frame in TARGETS.items():
            frames = videos[sequence]
            native_rows = {r['frame']: r for r in json.loads(
                (args.run_dir / 'native' / f'{sequence}_frames.json').read_text())}
            regional_rows = {r['frame']: r for r in json.loads(
                (args.run_dir / 'regional' / f'{sequence}_frames.json').read_text())}
            if target_frame not in native_rows or target_frame not in regional_rows:
                raise RuntimeError(f'Missing source target frame for {sequence}/{target_frame}')
            if not regional_rows[target_frame]['present'] or regional_rows[target_frame]['scores']['dice'] != 0:
                raise RuntimeError(f'Frame is not a regional zero-Dice present target: {sequence}/{target_frame}')
            model = build_author_predictor(args, repo)
            model.requires_grad_(False)
            with tempfile.TemporaryDirectory(prefix=f'{sequence}_', dir=args.output_dir) as temp, torch.inference_mode():
                state, prompt = prepare(model, frames, Path(temp))
                if target_frame <= prompt:
                    raise RuntimeError('Target is not after the prompt')
                found = False
                for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                    if frame == prompt:
                        continue
                    row = native_rows.get(frame)
                    if ids != [1] or row is None or tensor_hash(logits) != row['prediction_sha256']:
                        raise RuntimeError(f'Native trajectory mismatch at {sequence}/{frame}')
                    if frame < target_frame:
                        continue
                    if frame != target_frame:
                        raise RuntimeError('Propagation skipped the diagnostic target')
                    found = True
                    sample = capture_training_read(model, state, frame)
                    stored = state['output_dict_per_obj'][0]['non_cond_frame_outputs'].get(frame)
                    if stored is None:
                        stored = state['output_dict_per_obj'][0]['cond_frame_outputs'].get(frame)
                    if stored is None:
                        raise RuntimeError('Native target output was not stored')
                    gt = torch.from_numpy((load_label_mask(frames[frame].mask_path, [1]) == 1).copy())[None, None].float()
                    target = F.interpolate(gt, size=stored['pred_masks'].shape[-2:], mode='nearest')[0, 0].numpy() > 0
                    native_bank = fingerprint(state['output_dict_per_obj'])['sha256']
                    sample_bank = fingerprint(sample['output_dict'])['sha256']
                    modes = {}
                    for condition in ('zero', 'regional'):
                        heads = QueryConditionedRegionHeads().eval()
                        if condition == 'regional':
                            head_path = args.run_dir / 'regional' / 'head.pt'
                            head_receipt = json.loads((args.run_dir / 'regional' / 'training.json').read_text())
                            heads.load_state_dict(load(head_path, head_receipt['checkpoint_sha256']), strict=True)
                        with NativeRegionalReadout(model, heads, 'regional', sequence, seed) as observer:
                            output = model.track_step(
                                frame_idx=frame, is_init_cond_frame=False,
                                current_vision_feats=sample['vision'], current_vision_pos_embeds=sample['positions'],
                                feat_sizes=sample['sizes'], point_inputs=None, mask_inputs=None,
                                output_dict=sample['output_dict'], num_frames=sample['num_frames'],
                                track_in_reverse=False, run_mem_encoder=False)
                        if [call['layer'] for call in observer.calls] != [0, 1, 2, 3]:
                            raise RuntimeError('Fixed-history read missed an attention layer')
                        if condition == 'zero' and (
                            not torch.equal(output['pred_masks'], stored['pred_masks']) or
                            not torch.equal(output['object_score_logits'], stored['object_score_logits'])):
                            raise RuntimeError('Zero-gate replay is not bit-identical to native')
                        prediction = output['pred_masks'][0, 0].detach().cpu().numpy() > 0
                        modes[condition] = {
                            'low_res_dice': float(calculate_scores(prediction, target)['dice']),
                            'object_score': float(output['object_score_logits'].item()),
                            'positive_pixels': int(prediction.sum()),
                            'prediction_sha256': tensor_hash(output['pred_masks']),
                            'attention_reads': observer.calls,
                        }
                        if fingerprint(sample['output_dict'])['sha256'] != sample_bank:
                            raise RuntimeError('Fixed-history read changed sampled native memory')
                    if fingerprint(state['output_dict_per_obj'])['sha256'] != native_bank:
                        raise RuntimeError('Fixed-history read changed native inference state')
                    reports[sequence] = {
                        'frame': frame, 'prompt_frame': prompt,
                        'native_rollout_dice': native_rows[frame]['scores']['dice'],
                        'regional_rollout_dice': regional_rows[frame]['scores']['dice'],
                        'native_rollout_prediction_sha256': native_rows[frame]['prediction_sha256'],
                        'regional_rollout_prediction_sha256': regional_rows[frame]['prediction_sha256'],
                        'native_memory_bank_sha256': native_bank, 'arms': modes,
                    }
                    break
                if not found:
                    raise RuntimeError(f'Native propagation did not reach {sequence}/{target_frame}')
                model.reset_state(state)
            del model
            write(args.output_dir / 'progress.json', {'complete_sequences': len(reports), 'current': sequence})

        for relative, expected in run['sources'].items():
            if digest(ROOT / relative) != expected:
                raise RuntimeError(f'Frozen source changed during replay: {relative}')
        write(args.output_dir / 'results.json', {
            'complete': True, 'diagnostic': 'Outcome-informed one-step fixed-native-history replay',
            'reports': reports, 'source_hashes_verified': len(run['sources'])})
        write(args.output_dir / 'completion.json', {
            'complete': True, 'targets': len(reports), 'native_zero_gate_bit_identical': True,
            'memory_unchanged': True, 'all_four_layers_read': True})
    except BaseException as error:
        write(args.output_dir / 'failure.json', {'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
