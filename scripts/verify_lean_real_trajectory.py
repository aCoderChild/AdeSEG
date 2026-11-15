#!/usr/bin/env python3
"""Compare Lean routing identity on an automatic-prompt PolypGen clip.

Engineering-only: uses a test-only identity refiner, archived automatic prompts,
and no labels. Requires actual physical memory compression to have occurred.
Writes only a small hash/trace receipt to a new Google Drive experiment folder.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import MethodType, SimpleNamespace

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest, stage_frames
from modeling.lean_contrast_refinement_integration import route_lean_contrast_refinement
from modeling.lean_rope_cpu_repeat_fix import corrected_cpu_key_frequency_repeat
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor, source_identity


class Identity(torch.nn.Module):
    _trained_checkpoint_loaded = True  # Test stimulus, never an experiment checkpoint.

    def forward(self, early, deep, logits):
        return logits.clone()


def thash(x):
    if x is None:
        return None
    if not torch.is_tensor(x):
        raise TypeError('Expected tensor')
    x = x.detach().cpu().contiguous()
    return hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'prompt-dir', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--sequence', default='seq4')
    p.add_argument('--end-frame', type=int, default=20)
    p.add_argument('--rope-mode', choices=('author_cpu', 'corrected_cpu_repeat'), default='author_cpu')
    a = p.parse_args()
    prompt_path = a.prompt_dir / a.sequence / 'prompt.json'
    prompt = json.loads(prompt_path.read_text())
    start = prompt['first_detection_index']
    if not isinstance(start, int) or start <= 0 or a.end_frame <= start:
        raise ValueError('Need a nonzero prompt followed by propagation')
    videos = {**load_manifest(ROOT/'data/polypgen_sequence.jsonl', 'dev'),
              **load_manifest(ROOT/'data/polypgen_sequence.jsonl', 'test')}
    frames = videos[a.sequence][:a.end_frame + 1]
    if len(frames) != a.end_frame + 1:
        raise ValueError('Insufficient source frames')
    metadata = json.loads((a.prompt_dir / a.sequence / 'frames.json').read_text())
    if any(digest(f.image_path) != metadata[i]['image_sha256'] for i, f in enumerate(frames)):
        raise ValueError('Source frames differ from automatic prompt provenance')
    out = validate_output(a.output)
    out.mkdir(parents=True, exist_ok=False)
    repo, rev = source_identity('lean')
    torch.set_num_threads(2)
    identity = Identity().eval()
    all_runs = {}
    with tempfile.TemporaryDirectory(prefix='lean_real_trajectory_') as temp:
        stage_frames(frames, Path(temp))
        for arm in ('baseline', 'A', 'B', 'C', 'D'):
            model = build_author_predictor(SimpleNamespace(method='lean', checkpoint=a.checkpoint, device='cpu'), repo)
            if model.mem_keep_ratio != 0.25:
                raise RuntimeError('Expected author pruning configuration')
            state = model.init_state(video_path=temp, offload_video_to_cpu=True,
                                     offload_state_to_cpu=False)
            compressions = []
            old_compress = model._compress_memory_entry

            def capture_compress(self, entry):
                pending = entry.get('pending_keep_idx')
                before = None if pending is None else thash(pending)
                result = old_compress(entry)
                if before is not None:
                    compressions.append({'frame': entry.get('frame_idx'), 'keep_sha256': before,
                                         'state': entry.get('smr_state'),
                                         'original': entry.get('smr_original_token_count'),
                                         'retained': entry.get('smr_pruned_token_count')})
                return result
            model._compress_memory_entry = MethodType(capture_compress, model)
            outputs, states = [], []
            ctx = nullcontext() if arm == 'baseline' else route_lean_contrast_refinement(model, identity, arm, test_only=True)
            from sam2.modeling import memory_attention
            rope = (nullcontext() if a.rope_mode == 'author_cpu' else
                    corrected_cpu_key_frequency_repeat(memory_attention))
            with torch.inference_mode(), rope, ctx:
                _, ids, logits = model.add_new_points_or_box(state, frame_idx=start, obj_id=1,
                                                            box=np.asarray(prompt['box'], np.float32))
                if ids != [1]:
                    raise RuntimeError('Unexpected prompt object')
                if prompt['points'] is not None:
                    _, ids, logits = model.add_new_points_or_box(state, frame_idx=start, obj_id=1,
                        points=np.asarray(prompt['points'], np.float32), labels=np.array([1,0], np.int32),
                        clear_old_points=False)
                initial = logits[0,0].cpu().numpy() > 0
                if not initial.any():
                    raise RuntimeError('Automatic initial mask is empty')
                model.add_new_mask(state, frame_idx=start, obj_id=1, mask=initial)
                for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=start, reverse=False):
                    if ids != [1]:
                        raise RuntimeError('Unexpected propagation object')
                    outputs.append({'frame': frame, 'sha256': thash(logits)})
                    entry = state['output_dict_per_obj'][0]['non_cond_frame_outputs'].get(frame)
                    if entry is not None:
                        states.append({'frame': frame, **{k: thash(entry[k]) for k in
                          ('pred_masks', 'maskmem_features', 'obj_ptr', 'object_score_logits')}})
            model.reset_state(state)
            del model
            all_runs[arm] = {'outputs': outputs, 'states_at_write': states, 'compressions': compressions}
            if not compressions or any(e['state'] != 'pruned' or not e['retained'] or e['retained'] >= e['original'] for e in compressions):
                raise RuntimeError(f'{arm}: physical memory compression was not confirmed')
            if arm != 'baseline' and all_runs[arm] != all_runs['baseline']:
                (out / 'mismatch.json').write_text(json.dumps(all_runs, indent=2) + '\n')
                raise RuntimeError(f'{arm}: identity trajectory differs; see mismatch receipt')
            print(f'PASS {arm}: {len(outputs)} frames, {len(compressions)} compressions', flush=True)
    result = {'passed': True, 'scope': 'test-only identity on a short automatic-prompt real video; no efficacy',
              'sequence': a.sequence, 'start_frame': start, 'end_frame': a.end_frame,
              'checkpoint_sha256': digest(a.checkpoint), 'prompt_sha256': digest(prompt_path),
              'author_revision': rev, 'runner_sha256': digest(Path(__file__)),
              'rope_mode': a.rope_mode, 'rope_note': 'corrected_cpu_repeat does not address compressed-token or query rotary positions',
              'routing_sha256': digest(ROOT/'modeling/lean_contrast_refinement_integration.py'),
              'runs': all_runs}
    (out/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
