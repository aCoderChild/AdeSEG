"""Gate query-conditioned regional attention on one real predicted-memory clip."""
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
from modeling.native_region_readout import NativeRegionalReadout
from modeling.paired_readout_training import clip_loss
from modeling.query_conditioned_region_readout import QueryConditionedRegionHeads
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import write
from scripts.verify_on_demand_clip import capture_clip

SOURCE_FILES = [
    'protocols/query_conditioned_region_readout_polypgen_v1.json',
    'scripts/verify_query_conditioned_region_readout.py',
    'scripts/run_multiclip_readout_pilot.py',
    'scripts/verify_on_demand_clip.py',
    'scripts/run_cfbi_region_pilot.py',
    'scripts/run_author_baseline.py',
    'scripts/memory_token_io.py',
    'modeling/query_conditioned_region_readout.py',
    'modeling/region_mass_attention.py',
    'modeling/native_region_readout.py',
    'modeling/paired_readout_training.py',
    'modeling/cfbi_region_gate.py',
    'modeling/cfbi_training.py',
    'modeling/decoder_training_mask.py',
    'modeling/regional_refinement.py',
    'modeling/region_definitions.py',
    'modeling/region_tokens.py',
    'datasets/sam2_clip_sampler.py',
    'datasets/video.py',
    'datasets/io.py',
    'tests/test_query_conditioned_region_readout.py',
    'tests/test_region_mass_attention.py',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sequence', default='seq5')
    parser.add_argument('--seed', type=int, default=20261104)
    parser.add_argument('--length', type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        protocol = json.loads((ROOT / SOURCE_FILES[0]).read_text())
        manifest = ROOT / protocol['data']['manifest']
        if digest(args.checkpoint) == '':
            raise RuntimeError('Missing checkpoint')
        frames = load_manifest(manifest, 'dev')[args.sequence]
        clip = sample_clip(frames, args.length, args.seed)
        repo, revision = activate_author_repo()
        model = build_author_predictor(SimpleNamespace(
            method='sam2', device='cpu', model_config=protocol['model_config'],
            checkpoint=args.checkpoint), repo)
        model.requires_grad_(False)

        samples = capture_clip(model, clip, args.output_dir)
        if len(samples) != args.length - 1:
            raise RuntimeError('Native clip capture is incomplete')

        results = {}
        for mode in ('regional', 'shuffled'):
            heads = QueryConditionedRegionHeads().train()
            optimizer = torch.optim.Adam(heads.parameters(), lr=protocol['training']['learning_rate'])
            bank_hashes = [fingerprint(sample['output_dict'])['sha256'] for sample in samples]
            optimizer.zero_grad()
            loss, calls = clip_loss(model, heads, samples, mode, args.sequence,
                                    check_native=True, shuffle_seed=args.seed,
                                    return_trace=True)
            if len(calls) != 4 * len(samples) or [r['layer'] for r in calls] != [0, 1, 2, 3] * len(samples):
                raise RuntimeError('Expected all four attention layers on every clip frame')
            loss.backward()
            gradients = [float(layer.query_gate.grad.abs().sum())
                         if layer.query_gate.grad is not None else 0.0 for layer in heads.layers]
            if not torch.isfinite(loss) or not all(torch.isfinite(torch.tensor(g)) and g > 0 for g in gradients):
                raise RuntimeError('Expected finite nonzero query-gate gradients in all layers')
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError('Frozen SAM2 acquired gradients')
            if bank_hashes != [fingerprint(sample['output_dict'])['sha256'] for sample in samples]:
                raise RuntimeError('Training changed captured native memory')
            optimizer.step()
            if not all(torch.isfinite(layer.query_gate).all() and layer.query_gate.count_nonzero() > 0
                       for layer in heads.layers):
                raise RuntimeError('Optimizer did not produce finite active gates')

            heads.eval()
            native_logits = {sample['frame']: sample['native_logits'] for sample in samples}
            rows = []
            with tempfile.TemporaryDirectory(prefix='query_region_eval_', dir=args.output_dir) as temp:
                state, prompt = prepare(model, clip, Path(temp))
                with NativeRegionalReadout(model, heads, mode, args.sequence, args.seed) as observer, torch.inference_mode():
                    for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                        if ids != [1]:
                            raise RuntimeError('Unexpected object IDs during active propagation')
                        if frame == prompt:
                            continue
                        read = observer.observer.trace[-1]
                        if read['frame'] != frame or read['layers'] != [0, 1, 2, 3]:
                            raise RuntimeError('Active propagation missed an attention layer')
                        native = native_logits[frame]
                        rows.append({'frame': frame, 'prediction_sha256': digest_tensor(logits),
                                     'differs_from_native': not torch.equal(
                                         logits.detach().cpu().float(), native.detach().cpu().float()),
                                     'read': read})
                model.reset_state(state)
            results[mode] = {'loss': float(loss.detach()), 'gradient_l1': gradients,
                             'active_frames': rows}

        sources = {name: digest(ROOT / name) for name in SOURCE_FILES}
        completion = {'passed': True, 'sequence': args.sequence,
                      'manifest_frames': [frame.frame_index for frame in clip],
                      'checkpoint_sha256': digest(args.checkpoint),
                      'manifest_sha256': digest(manifest), 'sam2_revision': revision,
                      'seed': args.seed, 'modes': results, 'sources': sources,
                      'scope': 'one real five-frame clip, zero-gate identity and one optimizer update; engineering gate only'}
        write(args.output_dir / 'completion.json', completion)
    except BaseException as error:
        write(args.output_dir / 'failure.json', {'error': repr(error)})
        raise


def digest_tensor(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()


if __name__ == '__main__':
    main()
