"""Gate zero and trained regional readouts on a real PolypGen trajectory."""
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
from modeling.cfbi_training import object_presence_loss
from modeling.native_region_readout import NativeRegionalReadout, RegionalReadoutHeads
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import load, write

SOURCES = ['scripts/verify_readout_propagation.py', 'modeling/native_region_readout.py',
           'modeling/region_readout.py', 'modeling/cfbi_region_gate.py', 'modeling/cfbi_training.py']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'pilot-dir', 'native-frames', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--mode', choices=('regional', 'shuffled', 'generic'), default='regional')
    p.add_argument('--train-one-step', action='store_true')
    p.add_argument('--max-frames', type=int, default=0)
    args = p.parse_args()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(20261031)
        sources = {path: digest(ROOT / path) for path in SOURCES}
        setup = json.loads((args.pilot_dir / 'setup.json').read_text())
        if digest(args.checkpoint) != setup['checkpoint_sha256']:
            raise RuntimeError('Checkpoint mismatch')
        expected = {r['frame']: r['prediction_sha256'] for r in json.loads(args.native_frames.read_text())}
        repo, revision = activate_author_repo()
        model = build_author_predictor(SimpleNamespace(method='sam2', device='cpu', checkpoint=args.checkpoint), repo)
        model.requires_grad_(False)
        heads = RegionalReadoutHeads()
        loss_value = None
        if args.train_one_step:
            row = setup['samples'][12]
            sample = load(row['path'], row['sha256'])
            optimizer = torch.optim.Adam(heads.parameters(), lr=0.001)
            with NativeRegionalReadout(model, heads, args.mode, row['sequence']):
                output = model.track_step(frame_idx=sample['frame'], is_init_cond_frame=False,
                    current_vision_feats=sample['vision'], current_vision_pos_embeds=sample['positions'],
                    feat_sizes=sample['sizes'], point_inputs=None, mask_inputs=None,
                    output_dict=sample['output_dict'], num_frames=sample['num_frames'],
                    track_in_reverse=False, run_mem_encoder=False)
            loss = object_presence_loss(output['object_score_logits'], sample['target'])
            loss.backward()
            if not all(x.mix.weight.grad is not None and torch.isfinite(x.mix.weight.grad).all() for x in heads.layers):
                raise RuntimeError('Invalid training gradients')
            optimizer.step()
            if not all(torch.isfinite(x.mix.weight).all() and x.mix.weight.count_nonzero() > 0 for x in heads.layers):
                raise RuntimeError('Training step did not produce finite nonzero weights')
            loss_value = float(loss.detach())
        heads.eval()
        videos = {**load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'dev'),
                  **load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'test')}
        rows = []
        with tempfile.TemporaryDirectory(prefix='frames_', dir=args.output_dir) as temp:
            state, prompt = prepare(model, videos['seq2'], Path(temp))
            with NativeRegionalReadout(model, heads, args.mode, 'seq2') as adapter, torch.inference_mode():
                for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                    if ids != [1]:
                        raise RuntimeError('Unexpected object IDs')
                    if frame == prompt:
                        continue
                    if frame not in expected:
                        raise RuntimeError('Missing native frame')
                    if not torch.isfinite(logits).all():
                        raise RuntimeError(f'Nonfinite logits at frame {frame}')
                    calls = adapter.calls[-4:]
                    if [c['layer'] for c in calls] != [0, 1, 2, 3] or adapter.observer.trace[-1]['frame'] != frame:
                        raise RuntimeError('Incomplete cross-attention read')
                    prediction_hash = tensor_hash(logits)
                    if not args.train_one_step and prediction_hash != expected[frame]:
                        raise RuntimeError(f'Native prediction mismatch at frame {frame}')
                    rows.append({'frame': frame, 'prediction_sha256': prediction_hash,
                                 'native_identical': prediction_hash == expected[frame], 'read': calls})
                    write(args.output_dir / 'progress.json', {'frame': frame, 'checked_frames': len(rows)})
                    if args.max_frames and len(rows) >= args.max_frames:
                        break
            model.reset_state(state)
        if not rows or (not args.max_frames and len(rows) != len(expected)):
            raise RuntimeError('Trajectory coverage incomplete')
        if sources != {path: digest(ROOT / path) for path in SOURCES}:
            raise RuntimeError('Source changed during gate')
        write(args.output_dir / 'completion.json', {'passed': True, 'mode': args.mode, 'trained_one_step': args.train_one_step,
              'sample_training_loss': loss_value, 'sequence': 'seq2', 'sam2_revision': revision, 'frames': rows,
              'sources': sources,
              'scope': 'engineering check; not evidence of segmentation improvement'})
    except BaseException as error:
        write(args.output_dir / 'failure.json', {'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
