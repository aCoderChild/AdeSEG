"""Capture consecutive native reads for the regional training feasibility pilot."""
import argparse
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from modeling.cfbi_training import capture_training_read
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import CORE, activate_author_repo, core_hashes, native_reference, prepare
from scripts.run_regional_refinement import save_tensor, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'reference-dir', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    out = validate_output(a.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    manifest = ROOT / 'data/polypgen_sequence.jsonl'
    videos = {**load_manifest(manifest, 'dev'), **load_manifest(manifest, 'test')}
    repo, revision = activate_author_repo()
    pairs = []
    for name in ('seq4', 'seq5', 'seq9'):
        model = build_author_predictor(SimpleNamespace(method='sam2', device='cpu', checkpoint=a.checkpoint), repo)
        reference = native_reference(a.reference_dir, name)
        with tempfile.TemporaryDirectory(prefix='frames_', dir=out) as temp, torch.inference_mode():
            state, prompt = prepare(model, videos[name], Path(temp))
            selected = {prompt + 1, prompt + 2}
            if max(selected) >= len(videos[name]):
                raise RuntimeError(f'{name} is too short')
            samples = []
            for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=prompt):
                core, stored = core_hashes(state, frame, logits)
                if ids != [1] or frame not in reference:
                    raise RuntimeError(f'Unexpected frame at {name}/{frame}')
                mismatches = [k for k in CORE if core[k] != reference[frame][k]]
                if mismatches:
                    raise RuntimeError(f'Native integrity mismatch at {name}/{frame}: {mismatches}')
                if frame in selected:
                    sample = capture_training_read(model, state, frame)
                    sample['native_logits'] = stored['pred_masks'].detach().float().clone()
                    sample['target'] = torch.from_numpy((load_label_mask(videos[name][frame].mask_path, [1]) == 1).copy())[None, None].float()
                    path = out / f'{name}_{frame:06d}.pt'
                    save_tensor(sample, path)
                    samples.append({'frame': frame, 'path': str(path), 'sha256': digest(path), 'native': core})
                if frame == prompt + 2:
                    break
            if [row['frame'] for row in samples] != sorted(selected):
                raise RuntimeError(f'Incomplete pair for {name}')
            pairs.append({'sequence': name, 'prompt': prompt, 'samples': samples})
            model.reset_state(state)
    write(out / 'pairs.json', {'pairs': pairs, 'sam2_revision': revision,
          'checkpoint_sha256': digest(a.checkpoint), 'manifest_sha256': digest(manifest),
          'scope': 'exploratory training inputs; no optimization or evaluation'})


if __name__ == '__main__':
    main()
