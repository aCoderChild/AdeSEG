"""Trace decoder presence scores and memory writes at two regional-run failures."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from modeling.cutie_region_projection import CutieRegionProjectionHeads
from modeling.native_region_readout import NativeRegionalReadout
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import load, write


def _stat(tensor):
    if tensor is None:
        return None
    value = tensor.detach().float()
    return {
        'shape': list(value.shape),
        'min': float(value.min()),
        'max': float(value.max()),
        'mean': float(value.mean()),
        'positive_fraction': float((value > 0).float().mean()),
        'l2_norm': float(value.norm()),
    }


def trace_sequence(args, repo, protocol, frames, sequence, target, heads=None):
    model = build_author_predictor(args, repo)
    model.requires_grad_(False)
    records = {}
    active_frame = [None]

    def decoder_hook(_module, _inputs, output):
        masks, _ious, _tokens, score = output
        records.setdefault(active_frame[0], {})['raw_decoder_mask'] = _stat(masks[:, 0])
        records[active_frame[0]]['decoder_object_score'] = _stat(score)

    decoder_handle = model.sam_mask_decoder.register_forward_hook(decoder_hook)
    original_track_step = model.track_step

    def traced_track_step(*call_args, **kwargs):
        frame = kwargs.get('frame_idx', call_args[0] if call_args else None)
        active_frame[0] = int(frame)
        output = original_track_step(*call_args, **kwargs)
        row = records.setdefault(active_frame[0], {})
        row['output_object_score'] = _stat(output['object_score_logits'])
        row['stored_mask'] = _stat(output['pred_masks'])
        row['stored_memory'] = _stat(output['maskmem_features'])
        return output

    model.track_step = traced_track_step
    with tempfile.TemporaryDirectory(prefix=f'{sequence}_', dir=args.output_dir) as temp:
        state, prompt = prepare(model, frames, Path(temp))
        if target <= prompt:
            raise ValueError(f'{sequence}: target must follow prompt')
        context = (NativeRegionalReadout(model, heads, 'regional', sequence)
                   if heads is not None else None)
        manager = context if context is not None else _NullContext()
        with manager, torch.inference_mode():
            for frame, ids, logits in model.propagate_in_video(
                    state, start_frame_idx=prompt, max_frame_num_to_track=target - prompt):
                if ids != [1]:
                    raise RuntimeError(f'Unexpected object IDs in {sequence}/{frame}')
                if frame == prompt:
                    continue
                row = records.get(frame)
                if row is None:
                    raise RuntimeError(f'Missing decoder trace for {sequence}/{frame}')
                row['propagated_mask'] = _stat(logits)
                row['regional_attention_layers'] = (
                    context.calls[-4:] if context is not None else None)
                if frame == target:
                    break
        model.reset_state(state)
    decoder_handle.remove()
    model.track_step = original_track_step
    del model
    return {'sequence': sequence, 'prompt': prompt, 'target': target,
            'records': {str(frame): records[frame] for frame in sorted(records)
                        if prompt < frame <= target}}


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads((ROOT / 'protocols/cutie_region_projection_polypgen_v1.json').read_text())
    training = json.loads((args.run_dir / 'training_summary.json').read_text())
    if digest(args.checkpoint) != json.loads((args.run_dir / 'setup.json').read_text())['checkpoint_sha256']:
        raise RuntimeError('Checkpoint differs from the completed regional run')
    heads = CutieRegionProjectionHeads().eval()
    heads.load_state_dict(load(args.run_dir / 'regional_summary_heads.pt',
                               training['checkpoint_sha256']), strict=True)
    repo, revision = activate_author_repo()
    args.method, args.device = 'sam2', 'cpu'
    args.model_config = protocol['model']['sam2_configuration']
    videos = {**load_manifest(ROOT / protocol['data']['manifest'], 'dev'),
              **load_manifest(ROOT / protocol['data']['manifest'], 'test')}
    traces = []
    for sequence, target in [('seq16', 13), ('seq18', 5)]:
        frames = videos[sequence]
        traces.append({
            'native': trace_sequence(args, repo, protocol, frames, sequence, target),
            'regional': trace_sequence(args, repo, protocol, frames, sequence, target, heads),
        })
    write(args.output_dir / 'collapse_diagnostic.json', {
        'sam2_revision': revision,
        'training_head_sha256': training['checkpoint_sha256'],
        'purpose': 'Compare decoder object-presence score, pre-suppression decoder mask, stored mask and memory at first severe regional trajectory collapse.',
        'traces': traces,
    })
    write(args.output_dir / 'completion.json', {
        'complete': True,
        'sequences': ['seq16', 'seq18'],
        'targets': [13, 5],
        'conditions': ['native', 'regional_projection'],
        'scope': 'Targeted causal diagnostic only; no model weights or experiment scores changed.',
    })


if __name__ == '__main__':
    main()
