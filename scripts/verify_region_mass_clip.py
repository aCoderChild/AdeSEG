"""Verify normalized regional memory reweighting on one real SAM2 clip."""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from modeling.native_region_readout import RegionalMassHeads
from modeling.paired_readout_training import clip_loss
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_regional_refinement import write
from scripts.verify_on_demand_clip import capture_clip


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--sequence', default='seq4')
    p.add_argument('--seed', type=int, default=20261101)
    p.add_argument('--length', type=int, default=5)
    # The available MedSAM2_latest checkpoint has Hiera-T weights (12 trunk
    # blocks). Keep this in sync with its architecture; strict loading is used.
    p.add_argument('--model-config', default='configs/sam2.1/sam2.1_hiera_t.yaml')
    args = p.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        manifest = ROOT / 'data/polypgen_sequence.jsonl'
        videos = load_manifest(manifest, 'dev')
        frames = sample_clip(videos[args.sequence], args.length, args.seed)
        repo, revision = activate_author_repo()
        model = build_author_predictor(
            argparse.Namespace(method='sam2', device='cpu', checkpoint=args.checkpoint,
                               model_config=args.model_config), repo
        )
        model.requires_grad_(False)
        samples = capture_clip(model, frames, out)
        rows = {}
        for mode in ('regional', 'shuffled'):
            heads = RegionalMassHeads().train()
            optimizer = torch.optim.Adam(heads.parameters(), lr=0.01)
            optimizer.zero_grad()
            loss = clip_loss(model, heads, samples, mode, args.sequence,
                             check_native=True, shuffle_seed=args.seed)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite zero-bias loss')
            loss.backward()
            gradients = [float(layer.log_bias.grad.abs().sum()) if layer.log_bias.grad is not None else 0.0
                         for layer in heads.layers]
            if not all(torch.isfinite(torch.tensor(x)) for x in gradients) or not any(x > 0 for x in gradients):
                raise RuntimeError('Missing or nonfinite regional-bias gradients')
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError('Frozen SAM2 acquired parameter gradients')
            optimizer.step()
            bounds = [float((layer.max_log_scale * torch.tanh(layer.log_bias.detach())).abs().max())
                      for layer in heads.layers]
            if not all(torch.isfinite(torch.tensor(x)) and x <= torch.log(torch.tensor(2.)).item() + 1e-6
                       for x in bounds):
                raise RuntimeError('Regional log-bias exceeded its bound')
            with torch.no_grad():
                trained_loss = clip_loss(model, heads, samples, mode, args.sequence, shuffle_seed=args.seed)
            if not torch.isfinite(trained_loss):
                raise RuntimeError('Nonfinite post-update clip loss')
            rows[mode] = {'initial_loss': float(loss.detach()), 'post_update_loss': float(trained_loss),
                          'gradient_l1_by_layer': gradients,
                          'log_bias_by_layer': [layer.log_bias.detach().tolist() for layer in heads.layers],
                          'absolute_log_scale_bound_by_layer': bounds}
            del heads, optimizer

        write(out / 'completion.json', {
            'passed': True, 'sequence': args.sequence,
            'manifest_frames': [f.frame_index for f in frames], 'seed': args.seed,
            'checkpoint_sha256': digest(args.checkpoint), 'manifest_sha256': digest(manifest),
            'sam2_revision': revision, 'model_config': args.model_config, 'modes': rows,
            'native_masks_and_memory_bit_identical_at_zero_bias': True,
            'pointer_attention_mass_preserved': 'unit-tested',
            'elapsed_seconds': time.monotonic() - start,
            'peak_rss_bytes_macos': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'scope': 'single real clip, one optimizer update per mode; no accuracy claim'
        })
    except BaseException as error:
        write(out / 'failure.json', {'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
