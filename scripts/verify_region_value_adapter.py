"""Verify native identity, gradients and nonzero propagation for region values."""
import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from modeling.cfbi_training import object_presence_loss
from modeling.native_region_readout import NativeRegionalReadout
from modeling.paired_readout_training import clip_loss
from modeling.region_value_adapter import RegionValueHeads
from scripts.memory_token_io import digest
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import write
from scripts.verify_on_demand_clip import capture_clip

FILES = ['protocols/region_value_adapter_polypgen_v1.json',
         'scripts/verify_region_value_adapter.py', 'scripts/run_region_value_adapter.py',
         'modeling/region_value_adapter.py', 'modeling/native_region_readout.py',
         'modeling/paired_readout_training.py', 'modeling/cfbi_training.py',
         'modeling/decoder_training_mask.py', 'modeling/regional_refinement.py',
         'modeling/region_definitions.py', 'modeling/region_tokens.py',
         'modeling/cfbi_region_gate.py', 'scripts/verify_on_demand_clip.py',
         'datasets/sam2_clip_sampler.py', 'datasets/video.py', 'datasets/io.py',
         'tests/test_region_value_adapter.py']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sequence', default='seq5')
    parser.add_argument('--seed', type=int, default=20261104)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(args.seed)
        protocol = json.loads((ROOT / FILES[0]).read_text())
        manifest = ROOT / protocol['data']['manifest']
        frames = sample_clip(load_manifest(manifest, 'dev')[args.sequence],
                             protocol['data']['clip_length'], args.seed)
        repo, revision = activate_author_repo()
        args.method, args.device = 'sam2', 'cpu'
        args.model_config = protocol['model']['configuration']
        model = build_author_predictor(args, repo)
        model.requires_grad_(False)
        samples = capture_clip(model, frames, args.output_dir)
        heads = RegionValueHeads().train()
        optimizer = torch.optim.Adam(heads.parameters(), lr=protocol['training']['learning_rate'])
        loss, calls = clip_loss(model, heads, samples, 'regional_value', args.sequence,
                                check_native=True, return_trace=True)
        expected = 4 * len(samples)
        if len(calls) != expected or [row['layer'] for row in calls] != [0, 1, 2, 3] * len(samples):
            raise RuntimeError('Zero-adapter replay missed an attention layer')
        loss.backward()
        gradients = [float(layer.role_embeddings.grad.abs().sum())
                     if layer.role_embeddings.grad is not None else 0.0 for layer in heads.layers]
        if not torch.isfinite(loss) or not all(torch.isfinite(torch.tensor(g)) and g > 0 for g in gradients):
            raise RuntimeError('Expected finite nonzero role-embedding gradient in every layer')
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError('Frozen SAM2 acquired gradients')
        optimizer.step()
        if not all(torch.isfinite(layer.role_embeddings).all() and layer.role_embeddings.count_nonzero() > 0
                   for layer in heads.layers):
            raise RuntimeError('Optimizer did not activate every layer')

        heads.eval()
        native_logits = {sample['frame']: sample['native_logits'] for sample in samples}
        rows = []
        with tempfile.TemporaryDirectory(prefix='region_value_gate_', dir=args.output_dir) as temp:
            state, prompt = prepare(model, frames, Path(temp))
            if prompt != 0:
                raise RuntimeError('Sample clip must be prompted at its first frame')
            with NativeRegionalReadout(model, heads, 'regional_value', args.sequence) as observer, torch.inference_mode():
                for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                    if ids != [1]:
                        raise RuntimeError('Unexpected tracked object IDs')
                    if frame == prompt:
                        continue
                    calls = observer.calls[-4:]
                    if [call['layer'] for call in calls] != [0, 1, 2, 3]:
                        raise RuntimeError(f'Active read missed layer at frame {frame}')
                    if not torch.isfinite(logits).all():
                        raise RuntimeError(f'Nonfinite output at frame {frame}')
                    rows.append({'frame': frame,
                                 'prediction_sha256': tensor_hash(logits),
                                 'differs_from_native': not torch.equal(
                                     logits.detach().cpu().float(), native_logits[frame].detach().cpu().float()),
                                 'attention': calls})
            model.reset_state(state)
        if len(rows) != len(samples) or not any(row['differs_from_native'] for row in rows):
            raise RuntimeError('Trained adapter did not complete a nonzero propagation')
        sources = {name: digest(ROOT / name) for name in FILES}
        write(args.output_dir / 'completion.json', {
            'passed': True, 'sequence': args.sequence,
            'frames': [frame.frame_index for frame in frames],
            'checkpoint_sha256': digest(args.checkpoint), 'manifest_sha256': digest(manifest),
            'sam2_revision': revision, 'zero_adapter_native_identity': True,
            'all_four_layers_gradient_l1': gradients,
            'trained_propagation': rows, 'sources': sources,
            'scope': 'one real five-frame clip, one optimizer update; engineering gate only'})
    except BaseException as error:
        write(args.output_dir / 'failure.json', {'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
