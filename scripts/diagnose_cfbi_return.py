"""Replay trained CFBI gates on the identical native seq13 memory at return."""
import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from modeling.cfbi_region_gate import NativeCFBIGate, RegionGate
from modeling.cfbi_training import capture_training_read
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, core_hashes, native_reference, prepare, CORE
from scripts.run_regional_refinement import load, save_tensor, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('checkpoint', 'reference-dir', 'pilot-dir', 'output-dir'):
        p.add_argument('--' + n, type=Path, required=True)
    args = p.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    setup = json.loads((args.pilot_dir / 'setup.json').read_text())
    if digest(args.checkpoint) != setup['checkpoint_sha256']:
        raise RuntimeError('Checkpoint differs from pilot')
    manifest = ROOT / 'data/polypgen_sequence.jsonl'
    if digest(manifest) != setup['manifest_sha256']:
        raise RuntimeError('Manifest differs from pilot')
    repo, revision = activate_author_repo()
    model = build_author_predictor(SimpleNamespace(method='sam2', device='cpu', checkpoint=args.checkpoint), repo)
    frames = {**load_manifest(manifest, 'dev'), **load_manifest(manifest, 'test')}['seq13']
    reference = native_reference(args.reference_dir, 'seq13')
    with tempfile.TemporaryDirectory(prefix='frames_', dir=out) as temp, torch.inference_mode():
        state, prompt = prepare(model, frames, Path(temp))
        for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
            core, stored = core_hashes(state, frame, logits)
            if ids != [1] or any(core[k] != reference[frame][k] for k in CORE):
                raise RuntimeError(f'Native integrity mismatch at {frame}')
            if frame != 66:
                continue
            sample = capture_training_read(model, state, frame)
            save_tensor(sample, out / 'sample.pt')
            before = fingerprint(state['output_dict_per_obj'])['sha256']
            rows = {}
            for mode in ('zero', 'regional', 'shuffled', 'generic'):
                gate = RegionGate().eval()
                if mode != 'zero':
                    head = args.pilot_dir / mode / 'head.pt'
                    recorded = json.loads((args.pilot_dir / mode / 'training.json').read_text())['checkpoint_sha256']
                    gate.load_state_dict(load(head, recorded), strict=True)
                with NativeCFBIGate(model, gate, 'regional' if mode == 'zero' else mode, 'seq13') as observer:
                    result = model.track_step(frame_idx=frame, is_init_cond_frame=False,
                        current_vision_feats=sample['vision'], current_vision_pos_embeds=sample['positions'],
                        feat_sizes=sample['sizes'], point_inputs=None, mask_inputs=None,
                        output_dict=sample['output_dict'], num_frames=sample['num_frames'],
                        track_in_reverse=False, run_mem_encoder=False)
                if len(observer.trace) != 1 or observer.trace[0]['layers'] != [0, 1, 2, 3]:
                    raise RuntimeError('Incomplete attention read')
                if mode == 'zero' and (not torch.equal(result['pred_masks'], stored['pred_masks']) or not torch.equal(result['object_score_logits'], stored['object_score_logits'])):
                    raise RuntimeError('Zero-gate replay differs from native')
                rows[mode] = {'low_res_sha256': tensor_hash(result['pred_masks']),
                              'object_score': float(result['object_score_logits'].item()),
                              'positive_pixels': int((result['pred_masks'] > 0).sum()), 'read': observer.trace[0]}
            if before != fingerprint(state['output_dict_per_obj'])['sha256']:
                raise RuntimeError('Replay mutated native memory bank')
            write(out / 'results.json', {'sequence': 'seq13', 'frame': frame, 'native_integrity': core,
                  'native_object_score': float(stored['object_score_logits'].item()), 'sam2_revision': revision,
                  'checkpoint_sha256': setup['checkpoint_sha256'], 'arms': rows})
            write(out / 'completion.json', {'complete': True, 'zero_replay_bit_identical': True, 'bank_unchanged': True})
            break
        else:
            raise RuntimeError('Target frame not reached')

if __name__ == '__main__':
    main()
