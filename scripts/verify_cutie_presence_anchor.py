"""Real-data identity, gradient and propagation gate for presence-anchored Cutie reads."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from modeling.cutie_region_projection import CutieRegionProjectionHeads
from modeling.region_presence_anchor import clip_loss_with_presence_anchor
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import write
from scripts.capture_presence_clip import capture_presence_clip

PROTOCOL = 'protocols/cutie_region_presence_anchor_polypgen_v1.json'
SOURCES = [PROTOCOL, 'scripts/verify_cutie_presence_anchor.py',
           'scripts/run_cutie_presence_anchor.py', 'scripts/capture_presence_clip.py',
           'modeling/region_presence_anchor.py', 'scripts/verify_cutie_region_projection.py',
           'scripts/run_cutie_region_projection.py', 'modeling/cutie_region_projection.py',
           'modeling/cutie_region_prototypes.py', 'modeling/cutie_memory.py',
           'modeling/cutie_region_queries.py', 'modeling/paired_readout_training.py',
           'modeling/native_region_readout.py', 'modeling/cfbi_region_gate.py',
           'modeling/cfbi_training.py', 'modeling/decoder_training_mask.py',
           'modeling/regional_refinement.py',
           'modeling/region_definitions.py', 'modeling/region_tokens.py',
           'scripts/verify_on_demand_clip.py', 'datasets/sam2_clip_sampler.py',
           'datasets/video.py', 'datasets/io.py',
           'external/Cutie/cutie/model/transformer/object_summarizer.py',
           'external/Cutie/cutie/model/transformer/transformer_layers.py',
           'external/Cutie/cutie/model/transformer/positional_encoding.py',
           'tests/test_cutie_region_prototypes.py', 'tests/test_cutie_region_projection.py']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sequence', default='seq5')
    args = parser.parse_args()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(20261106)
        protocol = json.loads((ROOT / PROTOCOL).read_text())
        manifest = ROOT / protocol['data']['manifest']
        frames = sample_clip(load_manifest(manifest, 'dev')[args.sequence],
                             protocol['data']['clip_length'], protocol['data']['seed'])
        repo, revision = activate_author_repo()
        args.method, args.device = 'sam2', 'cpu'
        args.model_config = protocol['model']['sam2_configuration']
        model = build_author_predictor(args, repo)
        model.requires_grad_(False)
        samples = capture_presence_clip(model, frames, args.output_dir)
        captured = {sample['frame']: tensor_hash(sample['native_logits']) for sample in samples}
        memory_hashes = [fingerprint(sample['output_dict'])['sha256'] for sample in samples]
        heads = CutieRegionProjectionHeads().train()
        optimizer = torch.optim.AdamW(heads.parameters(),
                                      lr=protocol['training']['learning_rate'],
                                      weight_decay=protocol['training']['weight_decay'])
        loss, calls, anchor0 = clip_loss_with_presence_anchor(
            model, heads, samples, args.sequence,
            anchor_weight=protocol['training']['presence_anchor_weight'],
            check_native=True, return_trace=True)
        if memory_hashes != [fingerprint(sample['output_dict'])['sha256'] for sample in samples]:
            raise RuntimeError('Zero-scale pass changed a captured native memory bank')
        if len(calls) != 4 * len(samples) or [row['layer'] for row in calls] != [0, 1, 2, 3] * len(samples):
            raise RuntimeError('Zero-scale replay did not observe all four layers on every read')
        loss.backward()
        first_grads = [head.output_projection.weight.grad for head in heads.layers]
        if any(grad is None or not torch.isfinite(grad).all() or grad.abs().sum() == 0
               for grad in first_grads):
            raise RuntimeError('A zero-initialized output projection did not receive a finite nonzero gradient')
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError('Frozen SAM2 received gradients')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss2, calls2, anchor1 = clip_loss_with_presence_anchor(
            model, heads, samples, args.sequence,
            anchor_weight=protocol['training']['presence_anchor_weight'],
            check_native=False, return_trace=True)
        loss2.backward()
        reader_grad = sum(float(p.grad.abs().sum()) for head in heads.layers
                          for p in head.reader.parameters() if p.grad is not None)
        if not torch.isfinite(loss2) or not reader_grad > 0:
            raise RuntimeError('Pretrained Cutie attention reader did not receive a finite nonzero gradient')
        heads.eval()
        rows = []
        with tempfile.TemporaryDirectory(prefix='cutie_region_summary_', dir=args.output_dir) as temp:
            state, prompt = prepare(model, frames, Path(temp))
            if prompt != 0:
                raise RuntimeError('Engineering sample did not start at its prompt frame')
            from modeling.native_region_readout import NativeRegionalReadout
            with NativeRegionalReadout(model, heads, 'regional', args.sequence) as observer, torch.inference_mode():
                for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                    if ids != [1]:
                        raise RuntimeError('Unexpected object IDs')
                    if frame == prompt:
                        continue
                    calls = observer.calls[-4:]
                    if [call['layer'] for call in calls] != [0, 1, 2, 3]:
                        raise RuntimeError(f'Propagation missed a memory-attention layer at frame {frame}')
                    if any(call['spatial_keys'] % 4096 or call['pointer_keys'] < 0 for call in calls):
                        raise RuntimeError(f'Invalid spatial or pointer-key count at frame {frame}')
                    if not torch.isfinite(logits).all():
                        raise RuntimeError(f'Nonfinite mask prediction at frame {frame}')
                    rows.append({'frame': frame, 'prediction_sha256': tensor_hash(logits),
                                 'differs_from_native': tensor_hash(logits) != captured[frame],
                                 'attention': calls})
            model.reset_state(state)
        if len(rows) != len(frames) - 1 or not any(row['differs_from_native'] for row in rows):
            raise RuntimeError('Trained regional branch did not alter a complete finite real propagation')
        for head in heads.layers:
            if not torch.isfinite(head.output_projection.weight).all():
                raise RuntimeError('Nonfinite regional output projection')
        source_hashes = {path: digest(ROOT / path) for path in SOURCES}
        write(args.output_dir / 'completion.json', {
            'passed': True, 'sequence': args.sequence,
            'frames': [frame.frame_index for frame in frames],
            'checkpoint_sha256': digest(args.checkpoint), 'manifest_sha256': digest(manifest),
            'sam2_revision': revision, 'zero_projection_native_identity': True,
            'native_memory_immutable': True,
            'first_update_projection_gradient_l1': [float(grad.abs().sum()) for grad in first_grads],
            'second_update_reader_gradient_l1': reader_grad,
            'native_identity_presence_anchor_loss': float(anchor0.detach()),
            'post_update_presence_anchor_loss': float(anchor1.detach()),
            'propagation': rows, 'sources': source_hashes,
            'scope': 'one real five-frame engineering clip; not an accuracy experiment'})
    except BaseException as error:
        write(args.output_dir / 'failure.json', {'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
