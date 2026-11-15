"""Compare region assignments on identical saved SAM2 memory reads."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.segmentation import calculate_scores
from datasets.io import load_label_mask
from datasets.video import load_manifest
from modeling.cfbi_training import replay_training_read
from modeling.cutie_memory import CutieRegionGate
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_regional_refinement import load, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'pilot-dir', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--seq13-replay', type=Path)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    setup = json.loads((args.pilot_dir / 'setup.json').read_text())
    if digest(args.checkpoint) != setup['checkpoint_sha256']:
        raise RuntimeError('Checkpoint mismatch')
    info = json.loads((args.pilot_dir / 'regional' / 'training.json').read_text())
    gate = CutieRegionGate().eval()
    gate.load_state_dict(load(args.pilot_dir / 'regional' / 'head.pt', info['checkpoint_sha256']))
    repo, _ = activate_author_repo()
    model = build_author_predictor(SimpleNamespace(method='sam2', device='cpu', checkpoint=args.checkpoint), repo)
    rows = []
    if args.seq13_replay:
        replay = args.seq13_replay
        if not json.loads((replay / 'completion.json').read_text())['regional_replay_bit_identical']:
            raise RuntimeError('Unverified trajectory')
        path = replay / 'sample.pt'
        cases = [{'sequence': 'seq13', 'frame': 66, 'path': str(path), 'sha256': digest(path)}]
    else:
        cases = [setup['samples'][index] for index in (0, 3, 5, 8, 12, 13)]
    for row in cases:
        sample = load(row['path'], row['sha256'])
        if args.seq13_replay:
            frames = {**load_manifest('data/polypgen_sequence.jsonl', 'dev'),
                      **load_manifest('data/polypgen_sequence.jsonl', 'test')}['seq13']
            target = torch.from_numpy((load_label_mask(frames[66].mask_path, [1]) == 1).copy())[None, None].float()
            sample['target'] = F.interpolate(target, size=(256, 256), mode='nearest')
        before = fingerprint(sample['output_dict'])['sha256']
        outputs, traces = {}, {}
        with torch.inference_mode():
            for mode in ('regional', 'shuffled', 'generic'):
                masks, score, trace = replay_training_read(model, gate, sample, mode, row['sequence'], return_object_score=True)
                target = sample['target'][0, 0].numpy() > 0
                dice = calculate_scores(masks[0, 0].numpy() > 0, target)['dice']
                outputs[mode] = {'logits': masks, 'score': float(score.item()), 'dice': float(dice)}
                traces[mode] = trace
        if before != fingerprint(sample['output_dict'])['sha256']:
            raise RuntimeError('Saved memory changed')
        if any(t != traces['regional'] for t in traces.values()):
            raise RuntimeError('Read layout differs')
        reference = outputs['regional']['logits']
        rows.append({'sequence': row['sequence'], 'frame': row['frame'], 'sample_sha256': row['sha256'],
                     'target_present': bool(sample['target'].gt(0).any()), 'read': traces['regional'],
                     'conditions': {mode: {'score': v['score'], 'dice': v['dice'],
                         'logit_max_abs_difference': float((v['logits'] - reference).abs().max()),
                         'mask_disagreement_pixels': int(((v['logits'] > 0) != (reference > 0)).sum())}
                         for mode, v in outputs.items()}})
    scope = ('post-hoc previously inspected seq13 regional-CFBI history, not an independent evaluation'
             if args.seq13_replay else 'post-hoc fixed native training reads, not autonomous propagation or independent evaluation')
    write(args.output, {'scope': scope,
                        'gate': 'presence-supervised regional', 'rows': rows})


if __name__ == '__main__':
    main()
