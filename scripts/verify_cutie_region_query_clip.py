"""Gate the Cutie query adapter on a real SAM2 PolypGen clip."""
import argparse
import json
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from modeling.cutie_region_queries import CUTIE_REVISION, CutieRegionQueryHeads
from modeling.paired_readout_training import clip_loss
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_regional_refinement import write
from scripts.verify_on_demand_clip import capture_clip

PROTOCOL = ROOT / 'protocols/cutie_region_query_gate.json'
MODEL_CONFIG = 'configs/sam2.1/sam2.1_hiera_t.yaml'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cutie-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sequence', default='seq4')
    parser.add_argument('--length', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261101)
    args = parser.parse_args()
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        manifest = ROOT / 'data/polypgen_sequence.jsonl'
        frames = sample_clip(load_manifest(manifest, 'dev')[args.sequence], args.length, args.seed)
        repo, sam2_revision = activate_author_repo()
        model = build_author_predictor(SimpleNamespace(
            method='sam2', device='cpu', checkpoint=args.checkpoint,
            model_config=MODEL_CONFIG), repo)
        model.requires_grad_(False)
        samples = capture_clip(model, frames, output)
        results = {}
        for mode in ('regional', 'shuffled', 'generic'):
            heads = CutieRegionQueryHeads(pretrained_checkpoint=args.cutie_checkpoint).train()
            optimizer = torch.optim.Adam(heads.parameters(), lr=1e-4)
            optimizer.zero_grad(set_to_none=True)
            loss, trace = clip_loss(model, heads, samples, mode, args.sequence,
                                   check_native=True, shuffle_seed=args.seed,
                                   return_trace=True)
            if not torch.isfinite(loss):
                raise RuntimeError(f'{mode}: nonfinite identity loss')
            loss.backward()
            scale_grad = [float(layer.scale.grad.abs()) if layer.scale.grad is not None else 0.0
                          for layer in heads.layers]
            if not all(torch.isfinite(torch.tensor(value)) and value > 0 for value in scale_grad):
                raise RuntimeError(f'{mode}: invalid residual-scale gradients')
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError('Frozen SAM2 received gradients')
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updated_loss, updated_trace = clip_loss(
                model, heads, samples, mode, args.sequence,
                shuffle_seed=args.seed, return_trace=True)
            updated_loss.backward()
            query_grad = [float(layer.region_queries.grad.abs().sum())
                          if layer.region_queries.grad is not None else 0.0
                          for layer in heads.layers]
            bridge_grad = sum(float(parameter.grad.abs().sum())
                              for parameter in heads.core.parameters()
                              if parameter.requires_grad and parameter.grad is not None)
            if not torch.isfinite(updated_loss) or bridge_grad <= 0 or not all(
                    torch.isfinite(torch.tensor(value)) and value > 0 for value in query_grad):
                raise RuntimeError(f'{mode}: invalid region-query/projection gradients after update')
            if any(parameter.requires_grad for parameter in heads.core.block.parameters()):
                raise RuntimeError('Pretrained Cutie query block must remain frozen in this gate')
            expected_layers = [layer for _ in samples for layer in range(4)]
            if [row['layer'] for row in trace] != expected_layers:
                raise RuntimeError(f'{mode}: missing attention layers during identity pass')
            layout = lambda rows: [(row['layer'], row['spatial_keys'], row['pointer_keys'])
                                    for row in rows]
            if layout(trace) != layout(updated_trace):
                raise RuntimeError(f'{mode}: native token/pointer layout changed')
            if any(row['spatial_keys'] <= 0 or row['pointer_keys'] < 0 for row in trace):
                raise RuntimeError(f'{mode}: invalid spatial or pointer count')
            results[mode] = {
                'trainable_parameters': sum(parameter.numel() for parameter in heads.parameters()
                                            if parameter.requires_grad),
                'zero_scale_loss': float(loss.detach()),
                'post_update_loss': float(updated_loss.detach()),
                'scale_grad_l1_by_layer': scale_grad,
                'region_query_grad_l1_by_layer': query_grad,
                'shared_bridge_grad_l1': bridge_grad,
                'scale_after_update': [float(layer.scale.detach()) for layer in heads.layers],
                'attention_trace': trace,
                'updated_attention_trace': updated_trace,
            }
            del heads, optimizer

        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError('Frozen SAM2 acquired gradients')
        write(output / 'completion.json', {
            'passed': True,
            'sequence': args.sequence,
            'manifest_frames': [frame.frame_index for frame in frames],
            'seed': args.seed,
            'checkpoint_sha256': digest(args.checkpoint),
            'cutie_checkpoint_sha256': digest(args.cutie_checkpoint),
            'manifest_sha256': digest(manifest),
            'protocol_sha256': digest(PROTOCOL),
            'sam2_revision': sam2_revision,
            'cutie_revision': CUTIE_REVISION,
            'model_config': MODEL_CONFIG,
            'parameters_per_arm': next(iter(results.values()))['trainable_parameters'],
            'native_prediction_and_memory_identity_at_zero_scale': True,
            'all_four_layers_observed': True,
            'pointer_tokens_excluded_from_region_summaries': True,
            'modes': results,
            'elapsed_seconds': time.monotonic() - started,
            'peak_rss_bytes_macos': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'scope': 'real three-frame clip; engineering gate only; no accuracy claim'
        })
    except BaseException as error:
        write(output / 'failure.json', {'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
