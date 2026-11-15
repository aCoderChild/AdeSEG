"""Capture development misses using the existing hash-checked native workflow."""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.video import load_manifest
from scripts.memory_token_io import digest, validate_output
from scripts.run_cfbi_region_pilot import activate_author_repo, training_capture
from scripts.run_regional_refinement import write
import scripts.run_cfbi_region_pilot as pilot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'pilot-dir', 'reference-dir', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    args.method, args.device = 'sam2', 'cpu'
    setup = json.loads((args.pilot_dir / 'setup.json').read_text())
    if digest(args.checkpoint) != setup['checkpoint_sha256']:
        raise RuntimeError('Checkpoint differs from original pilot')
    manifest = Path('data/polypgen_sequence.jsonl')
    if digest(manifest) != setup['manifest_sha256']:
        raise RuntimeError('Manifest differs from original pilot')
    if digest(args.reference_dir / 'full_seq4_C0.zip') != setup['training_reference_sha256']['seq4']:
        raise RuntimeError('Native reference differs from original pilot')
    repo, _ = activate_author_repo()
    frames = load_manifest(manifest, 'dev')['seq4']
    pilot.training_targets = lambda prompt, length: [24, 25]
    rows = training_capture(args, 'seq4', frames, repo)
    for row in rows:
        sample = torch.load(row['path'], map_location='cpu', weights_only=False)
        if not sample['target'].gt(0).any() or sample['native_logits'].gt(0).any():
            raise RuntimeError('Expected a present target with empty native prediction')
    write(args.output_dir / 'training_samples.json', rows)
    write(args.output_dir / 'completion.json', {'passed': True, 'sequence': 'seq4', 'frames': [24, 25],
          'scope': 'outcome-selected development examples, not validation'})


if __name__ == '__main__':
    main()
