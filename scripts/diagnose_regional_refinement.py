"""Read-only checkpoint diagnostics, reusing source32 model/loss/evaluation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from evaluation.segmentation import calculate_scores
from evaluation.davis2017.metrics import db_eval_boundary
from modeling.regional_refinement import RegionalRefinement, refinement_loss
from scripts.run_regional_refinement import batch, check_sources, load, write, ARMS
from scripts.memory_token_io import digest, validate_output


def donor_mapping(rows, patterns):
    """Outcome-independent cross-sequence mapping, preserving valid groups."""
    result = []
    for i, row in enumerate(rows):
        candidates = [j for j, other in enumerate(rows)
                      if other['sequence'] != row['sequence'] and patterns[j] == patterns[i]]
        result.append(candidates[i % len(candidates)] if candidates else None)
    return result


def paired_modes(head, state, inputs):
    """BN updates are discarded after each diagnostic batch, even on errors."""
    head.load_state_dict(state)
    head.eval()
    try:
        with torch.no_grad():
            evaluation = head(*inputs)
            head.train()
            training = head(*inputs)
        return evaluation, training
    finally:
        head.load_state_dict(state)
        head.eval()


def summary(rows):
    result = {}
    for arm in ARMS:
        sequences = {}
        for name in sorted({r['sequence'] for r in rows}):
            group = [r for r in rows if r['arm'] == arm and r['sequence'] == name]
            modes = sorted({r['mode'] for r in group})
            sequences[name] = {}
            for mode in modes:
                items = [r for r in group if r['mode'] == mode]
                present = [r for r in items if r['present']]
                sequences[name][mode] = {
                    'frames': len(items), 'present_frames': len(present),
                    'present_dice': float(np.mean([r['dice'] for r in present])) if present else None,
                    'false_alarms': sum(r['false_alarm'] for r in items),
                    'mean_loss': float(np.mean([r['loss'] for r in items])) if items and 'loss' in items[0] else None,
                }
                for metric in ('iou', 'boundary_f', 'mean_abs_logit_change'):
                    eligible = present if metric != 'mean_abs_logit_change' else items
                    if eligible and metric in eligible[0]:
                        sequences[name][mode][metric] = float(np.mean([r[metric] for r in eligible]))
        result[arm] = sequences
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gate', action='store_true')
    args = parser.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = ROOT/'protocols/refinement_diagnostics_v1.json'
    source_setup = json.loads((args.source/'setup.json').read_text())
    check_sources(source_setup['sources'])
    states = {a: load(args.source/f'{a}_20261031/head.pt') for a in ARMS}
    heads = {a: RegionalRefinement().eval() for a in ARMS}
    for a in ARMS:
        heads[a].load_state_dict(states[a])
    training = json.loads((args.source/'training_samples.json').read_text())
    index = json.loads((args.source/'cache_index.json').read_text())
    validation = sorted([r for r in index if int(r['sequence'][3:]) >= 16],
                        key=lambda r: (int(r['sequence'][3:]), r['frame']))
    if args.gate:
        training = training[:4]
        validation = [next(r for r in validation if r['sequence'] == n) for n in ('seq18', 'seq23')]
    setup = {'protocol': json.loads(protocol.read_text()), 'gate': args.gate,
             'source_setup_sha256': digest(args.source/'setup.json'),
             'code_sha256': digest(__file__), 'protocol_sha256': digest(protocol),
             'checkpoints': {a: digest(args.source/f'{a}_20261031/head.pt') for a in ARMS}}
    write(out/'setup.json', setup)
    try:
        contexts = []
        for i, row in enumerate(training + validation):
            if digest(row['path']) != row['sha256']:
                raise RuntimeError(f'Cache hash mismatch: {row["path"]}')
            if i >= len(training):
                sample = load(row['path'])
                contexts.append((sample['contexts'], sample['valid']))
            write(out/'progress.json', {'phase': 'verify_cache', 'done': i+1, 'total': len(training)+len(validation)})
        patterns = [tuple(v['regional'].tolist()) for _, v in contexts]
        donors = donor_mapping(validation, patterns)
        write(out/'mapping.json', [{'recipient': [r['sequence'], r['frame']],
                                   'donor': None if j is None else [validation[j]['sequence'], validation[j]['frame']],
                                   'valid': patterns[i]} for i, (r, j) in enumerate(zip(validation, donors))])
        train_rows = []
        for start in range(0, len(training), 4):
            group = training[start:start+4]
            for arm in ARMS:
                inputs, target = batch([r['path'] for r in group], arm)
                evaluation, training_output = paired_modes(heads[arm], states[arm], inputs)
                if not all(torch.equal(v, states[arm][k]) for k, v in heads[arm].state_dict().items()):
                    raise RuntimeError('Diagnostic altered head state')
                for mode, outputs in [('eval', evaluation), ('train', training_output)]:
                    for i, row in enumerate(group):
                        prediction = outputs[i:i+1]
                        score = calculate_scores(prediction[0,0].numpy()>0, target[i,0].numpy()>0)
                        train_rows.append({'arm': arm, 'mode': mode, 'sequence': row['sequence'], 'frame': row['frame'],
                                           'present': bool(target[i].any()), 'false_alarm': bool((prediction>0).any()) and not bool(target[i].any()),
                                           'dice': score['dice'], 'loss': float(refinement_loss(prediction, target[i:i+1]))})
            write(out/'progress.json', {'phase': 'training_modes', 'done': min(start+4, len(training)), 'total': len(training)})
        write(out/'training_frames.json', train_rows)
        write(out/'training_summary.json', summary(train_rows))
        saved = {a: json.loads((args.source/f'{a}_20261031/results.json').read_text())['frames'] for a in ARMS}
        validation_rows = []
        for i, row in enumerate(validation):
            truth = load_label_mask(Path(row['mask']), [1]) == 1
            for arm in ARMS:
                inputs, _ = batch([row['path']], arm)
                with torch.no_grad():
                    original = heads[arm](*inputs)
                    outputs = [('original', original)]
                    if donors[i] is not None:
                        donor_context, donor_valid = contexts[donors[i]]
                        changed = (inputs[0], donor_context[arm][None], donor_valid[arm][None], inputs[3], inputs[4])
                        outputs.append(('donor', heads[arm](*changed)))
                reference = next(r for r in saved[arm][row['sequence']] if r['frame'] == row['frame'])
                for mode, output in outputs:
                    if not torch.isfinite(output).all():
                        raise RuntimeError('Nonfinite diagnostic output')
                    predicted = F.interpolate(output, size=truth.shape, mode='bilinear', align_corners=False)[0,0].numpy()>0
                    scores = calculate_scores(predicted, truth)
                    scores['boundary_f'] = float(db_eval_boundary(predicted, truth))
                    if mode == 'original' and any(abs(scores[k]-reference['scores'][k])>1e-9 for k in ('dice','iou','boundary_f')):
                        raise RuntimeError(f'Saved replay mismatch {arm}/{row}')
                    validation_rows.append({'arm': arm, 'mode': mode, 'sequence': row['sequence'], 'frame': row['frame'],
                                            'present': bool(truth.any()), 'false_alarm': bool(predicted.any()) and not bool(truth.any()),
                                            **scores, 'mean_abs_logit_change': float((output-original).abs().mean())})
            write(out/'progress.json', {'phase': 'context_swap', 'done': i+1, 'total': len(validation)})
            write(out/'validation_frames.json', validation_rows)
        write(out/'validation_summary.json', summary(validation_rows))
        lines = ['# Saved-head refinement diagnostics', '',
                 'Exploratory diagnostics only; no training or SAM2 inference.', '',
                 '| Arm | Sequence | Original present Dice | Donor present Dice |',
                 '|---|---|---:|---:|']
        for arm, sequences in summary(validation_rows).items():
            for name, modes in sequences.items():
                lines.append(f"| {arm} | {name} | {modes['original']['present_dice']} | {modes.get('donor', {}).get('present_dice')} |")
        lines += ['', 'Read training_summary.json for paired train/eval results and mapping.json for donor coverage.',
                  'Compare original/donor only on eligible matched frames when donors are unavailable. Context substitution is an out-of-distribution diagnostic.',
                  'Review these diagnostics before freezing any new training comparison; no automatic tuning or training launch.']
        (out/'results.md').write_text('\n'.join(lines)+'\n')
        check_sources(source_setup['sources'])
        if any(digest(args.source/f'{a}_20261031/head.pt') != setup['checkpoints'][a] for a in ARMS):
            raise RuntimeError('Checkpoint changed')
        write(out/'completion.json', {'complete': True, 'gate': args.gate, 'original_replay_passed': True,
                                     'train_samples': len(training), 'validation_samples': len(validation),
                                     'donor_eligible': sum(d is not None for d in donors), 'new_training_launched': False})
        write(out/'progress.json', {'phase': 'complete'})
    except BaseException as error:
        write(out/'failure.json', {'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
