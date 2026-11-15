"""Check native replay and training gradients for one sampled PolypGen clip."""
import argparse
import resource
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from modeling.cfbi_training import capture_training_read
from modeling.native_region_readout import RegionalReadoutHeads
from modeling.paired_readout_training import clip_loss
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import write


def capture_clip(model, frames, out):
    samples = []
    with tempfile.TemporaryDirectory(prefix='frames_', dir=out) as temp:
        with torch.inference_mode():
            state, prompt = prepare(model, frames, Path(temp))
            if prompt != 0:
                raise RuntimeError('Sampled first frame must be the prompt')
            for frame, ids, _ in model.propagate_in_video(state, start_frame_idx=prompt):
                if ids != [1]:
                    raise RuntimeError('Unexpected object IDs')
                if frame > 0:
                    sample = capture_training_read(model, state, frame)
                    stored = state['output_dict_per_obj'][0]['non_cond_frame_outputs'][frame]
                    with torch.inference_mode(False):
                        sample['native_logits'] = stored['pred_masks'].detach().float().clone()
                        sample['target'] = torch.from_numpy((load_label_mask(frames[frame].mask_path, [1]) == 1).copy())[None, None].float()
                    samples.append(sample)
            model.reset_state(state)
    if len(samples) != len(frames) - 1:
        raise RuntimeError('Missing sampled reads')
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sequence', default='seq4')
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--length', type=int, default=3)
    args = parser.parse_args()
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
        model = build_author_predictor(SimpleNamespace(method='sam2', device='cpu', checkpoint=args.checkpoint), repo)
        model.requires_grad_(False)
        samples = capture_clip(model, frames, out)
        results = {}
        for mode in ('regional', 'shuffled', 'generic'):
            heads = RegionalReadoutHeads().train()
            optimizer = torch.optim.Adam(heads.parameters(), lr=0.001)
            optimizer.zero_grad()
            loss = clip_loss(model, heads, samples, mode, args.sequence, check_native=True, shuffle_seed=args.seed)
            loss.backward()
            gradients = [float(layer.mix.weight.grad.abs().sum()) for layer in heads.layers]
            if not torch.isfinite(loss) or not all(torch.isfinite(torch.tensor(g)) and g > 0 for g in gradients):
                raise RuntimeError('Invalid gradients or loss')
            if any(p.grad is not None for p in model.parameters()):
                raise RuntimeError('Backbone gradients are not frozen')
            optimizer.step()
            if not all(torch.isfinite(layer.mix.weight).all() and layer.mix.weight.count_nonzero() > 0 for layer in heads.layers):
                raise RuntimeError('Invalid head update')
            results[mode] = {'loss': float(loss.detach()), 'gradient_l1': gradients}
        write(out / 'completion.json', {'passed': True, 'sequence': args.sequence,
              'manifest_frames': [f.frame_index for f in frames], 'seed': args.seed,
              'checkpoint_sha256': digest(args.checkpoint), 'manifest_sha256': digest(manifest),
              'sam2_revision': revision, 'modes': results, 'elapsed_seconds': time.monotonic() - start,
              'peak_rss_bytes_macos': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'scope': 'single clip, one optimizer update per arm; not an accuracy result'})
    except BaseException as error:
        write(out / 'failure.json', {'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
