#!/usr/bin/env python3
"""Verify all 27 frozen author videos before separately scoring labels."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from scripts.memory_token_io import digest, validate_output
from scripts.run_additional_author_comparator_v2 import (DATA, METHODS, REPRESENTATIVE, SEQUENCES,
    TRAINVAL_SEQUENCES, TRAIN_PREFLIGHT, identity, input_data, new_output, verify_gate)
from scripts.run_additional_author_comparator_study_v2 import (CHECKPOINT, DETECTOR, EXTENDED, OUT,
    REMAINING, commands, source_hashes)
from scripts.score_boundary_video_factorial_v1 import metric, summary, paired_summary
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.source_group_training_label_guard_v1 import mask_path_sets

PRIOR = {
    'extended': (OUT/'extended_boundary_factorial_v1_12ahgcx7',
                 OUT/'extended_boundary_factorial_v1_12ahgcx7_independent_audit_attempt01/audit.json'),
    'remaining': (OUT/'remaining_boundary_factorial_v1_a4_crb_n',
                  OUT/'remaining_boundary_factorial_v1_a4_crb_n_independent_audit_attempt01/audit.json'),
}


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def read_bound_receipt(path, expected):
    require(digest(path) == expected, 'Full-video receipt changed after launch')
    result = read(path)
    require(digest(path) == expected, 'Full-video receipt changed during read')
    return result


def install_phase_mask_guard(manifest):
    """Block Python-level mask opens until the verified scoring phase begins."""
    train, other = mask_path_sets(manifest)
    blocked = train | other
    state = {'active': True, 'blocked_paths': len(blocked), 'open_attempts': 0}
    def audit(event, args):
        if (not state['active'] or event != 'open' or not args or
                not isinstance(args[0], (str, bytes, os.PathLike))):
            return
        if os.path.realpath(os.fsdecode(args[0])) in blocked:
            state['open_attempts'] += 1
            raise PermissionError('Label open blocked before verified scoring')
    sys.addaudithook(audit)
    return state


def prior_data(name):
    root, audit_path = PRIOR[name]
    audit, scores, mapping = read(audit_path), read(root/'scores/results.json'), read(root/'run_map.json')
    auditor = ROOT / ('scripts/audit_remaining_boundary_factorial_v1.py' if name == 'remaining'
                      else 'scripts/audit_extended_boundary_factorial_v1.py')
    require(audit.get('passed') is True and audit.get('error_count') == 0 and
            audit.get('run') == str(root) and audit.get('score_sha256') == digest(root/'scores/results.json') and
            audit.get('run_map_sha256') == digest(root/'run_map.json') and
            audit.get('run_manifest_sha256') == digest(root/'run_manifest.json') and
            audit.get('audit_source_sha256') == digest(auditor) and scores.get('passed') is True,
            'Prior A/D factorial audit differs')
    return root, audit_path, audit, scores, mapping


def archived_frame_metadata(sequence):
    """Return a bound metadata path without parsing label/outcome fields."""
    if sequence in TRAINVAL_SEQUENCES:
        item = read(TRAIN_PREFLIGHT/'manifest.json')['sequences'][sequence]
        path = Path(item['source'])/sequence/'frames.json'
        expected = item['archived_frames_sha256']
    else:
        preflight = REMAINING if sequence in ('seq6', 'seq10') else EXTENDED
        item = read(preflight/'manifest.json')['sequences'][sequence]
        path = Path(item['archive_frames_path'])
        expected = item['archive_frames_sha256']
    require(digest(path) == expected, 'Archived frame metadata changed')
    return {'path': str(path.resolve()), 'sha256': expected}


def validate_frozen(run_map, *, allow_completed_verify=False, allow_completed_score=False):
    """Open no annotation masks; validate all current and prior predictions."""
    run_map = Path(run_map).resolve(strict=True)
    folder = run_map.parent
    require(folder.is_relative_to(OUT.resolve(strict=True)), 'Batch outside experiment root')
    launch_path, mapping = folder/'run_manifest.json', read(run_map)
    launch = read(launch_path)
    selection = tuple(launch.get('sequences', []))
    require(selection in (SEQUENCES, ('seq10',)), 'Comparator study scope differs')
    require(launch.get('run_map') == mapping
            and launch.get('methods') == list(METHODS)
            and launch.get('representative_for_method') == REPRESENTATIVE
            and launch.get('sources') == source_hashes() and launch.get('python_hash_seed') == '0'
            and launch.get('mps_fallback') == '0' and launch.get('mps_fast_math') == '0'
            and launch.get('checkpoint_sha256') == digest(CHECKPOINT)
            and launch.get('detector_sha256') == digest(DETECTOR)
            and launch.get('extended_preflight_sha256') == digest(EXTENDED/'manifest.json')
            and launch.get('remaining_preflight_sha256') == digest(REMAINING/'manifest.json')
            and launch.get('train_preflight_sha256') == digest(TRAIN_PREFLIGHT/'manifest.json'),
            'Frozen launch inputs differ')
    require(set(mapping) == set(selection) and all(set(mapping[s]) == set(METHODS) for s in selection),
            'Full prediction map differs')
    paths = [Path(mapping[s][m]).resolve(strict=True) for s in selection for m in METHODS]
    require(len(set(paths)) == len(selection) * len(METHODS) and all(p.is_relative_to(folder) for p in paths),
            'Prediction folders are aliased or outside the batch')
    planned, _ = commands(folder, selection)
    expected = [name for name, _ in planned if name not in ('verify', 'score', 'audit')]
    if allow_completed_verify or allow_completed_score:
        expected.append('verify')
    if allow_completed_score:
        expected.append('score')
    phases = [json.loads(x) for x in (folder/'phases.jsonl').read_text().splitlines() if x.strip()]
    require([p.get('phase') for p in phases] == expected and all(
        p.get('passed') is True and p.get('log_sha256') == digest(folder/(p['phase']+'.log')) for p in phases),
        'Every short gate and full-video phase must finish before verification')
    frozen_receipts = {p['phase']: p.get('receipt_sha256') for p in phases if p['phase'].startswith('full_')}
    require(len(frozen_receipts) == len(selection) * len(METHODS) and
            all(isinstance(x, str) and len(x) == 64 for x in frozen_receipts.values()),
            'Full-video receipt identities are missing from phase log')
    prior = {name: prior_data(name) for name in PRIOR}
    receipts, references, inputs = {}, {}, {}
    for seq in selection:
        frames, start, _, prompt = input_data(seq, EXTENDED, REMAINING, CHECKPOINT, DETECTOR)
        inputs[seq] = {'prompt_frame': start, 'prompt': prompt, 'frame_metadata': archived_frame_metadata(seq),
                       'images': [digest(frame.image_path) for frame in frames]}
        receipts[seq] = {}
        for method in METHODS:
            representative = REPRESENTATIVE[method]
            _, _, _, rep_prompt = input_data(representative, EXTENDED, REMAINING, CHECKPOINT, DETECTOR)
            root = Path(mapping[seq][method]).resolve(strict=True)
            receipt_path = root/'receipt.json'
            expected_receipt = frozen_receipts[f'full_{seq}_{method}']
            rec = read_bound_receipt(receipt_path, expected_receipt)
            gate = folder/'short'/method/representative/'gate/gate.json'
            expected_identity = identity(method, seq, CHECKPOINT, DETECTOR, EXTENDED, REMAINING, 'mps')
            checked = verify_gate(gate, expected_identity, rep_prompt, CHECKPOINT, DETECTOR, EXTENDED, REMAINING)
            require(rec.get('passed') is True and rec.get('full_video') is True and
                    rec.get('method') == method and rec.get('sequence') == seq and
                    rec.get('prompt_frame') == start and rec.get('prompt') == prompt and
                    rec.get('identity') == expected_identity and rec.get('state_sha256') == checked['state_sha256'] and
                    rec.get('gate_sha256') == digest(gate) and rec.get('representative_gate_sequence') == representative and
                    rec.get('noncausal') == (method == 'sam2long') and
                    rec.get('label_guard', {}).get('open_attempts') == 0,
                    'Full comparator receipt differs')
            records = rec.get('records', [])
            require(len(records) == len(frames) and [r['frame'] for r in records] == list(range(len(frames))),
                    'Comparator output frame coverage differs')
            for i, (frame, row) in enumerate(zip(frames, records, strict=True)):
                require(row.get('source_frame_index') == frame.frame_index and
                        row.get('image_sha256') == inputs[seq]['images'][i] and
                        row.get('post_prompt') == (i >= start), 'Comparator input identity differs')
                file = root/f'{i:06d}.png'
                require(digest(file) == row['mask_sha256'], 'Comparator mask changed')
                with Image.open(frame.image_path) as source, Image.open(file) as image:
                    values = np.asarray(image.convert('L'))
                    require(image.format == 'PNG' and image.size == source.size and np.isin(values, (0, 255)).all(),
                            'Comparator mask geometry or binary encoding differs')
                    if i < start:
                        require(not values.any(), 'Comparator pre-prompt output is not empty')
            receipts[seq][method] = {'path': str(root/'receipt.json'), 'sha256': digest(root/'receipt.json')}
        if seq in TRAINVAL_SEQUENCES:
            # No full-video A/D refiner rollout exists yet for these exposed
            # sequences. Score author systems now, without inventing a baseline.
            references[seq] = {}
            continue
        key = 'remaining' if seq in ('seq6', 'seq10') else 'extended'
        prior_root, audit_path, _, scores, prior_map = prior[key]
        references[seq] = {}
        for arm in ('A', 'D'):
            prior_folder = Path(prior_map[seq][arm]).resolve(strict=True)
            require(prior_folder.is_relative_to(prior_root.resolve(strict=True)), 'Prior output outside prior batch')
            completion = prior_folder/'completion.json'
            rec = read_bound_receipt(completion, scores['run_receipt_sha256'][seq][arm])
            require(digest(completion) == scores['run_receipt_sha256'][seq][arm] and
                    rec.get('passed') is True and rec.get('full_video') is True and rec.get('arm') == arm and
                    rec.get('sequence') == seq and rec.get('prompt_frame') == start and
                    rec.get('prompt_mask_sha256') == prompt['prompt_mask_sha256'] and
                    rec.get('checkpoint_sha256') == digest(CHECKPOINT) and
                    rec.get('detector_sha256') == digest(DETECTOR) and
                    rec.get('label_guard', {}).get('open_attempts') == 0,
                    'Prior matched Lean/refiner completion differs')
            require(len(rec['masks']) == len(frames) and
                    [row['frame'] for row in rec['masks']] == list(range(len(frames))), 'Prior mask coverage differs')
            for row in rec['masks']:
                require(digest(prior_folder/f"{row['frame']:06d}.png") == row['sha256'], 'Prior mask changed')
            references[seq][arm] = {'path': str(completion), 'sha256': digest(completion),
                                    'prior_audit_path': str(audit_path), 'prior_audit_sha256': digest(audit_path)}
    return {'selection': list(selection), 'run_map_sha256': digest(run_map), 'launch_manifest_sha256': digest(launch_path),
            'phase_log_sha256': digest(folder/'phases.jsonl'), 'receipts': receipts,
            'references': references, 'inputs': inputs}


def score(run_map, verification, output):
    out = new_output(output)
    guard = install_phase_mask_guard(DATA)
    run_map = Path(run_map).resolve(strict=True)
    folder = run_map.parent
    proof = read(verification)
    require(proof.get('passed') is True and proof.get('run_map_sha256') == digest(run_map)
            and proof.get('label_guard', {}).get('open_attempts') == 0
            and proof.get('scorer_sha256') == digest(Path(__file__)), 'Separate verification receipt differs')
    # Rehash all inputs before opening the first label; the separate verify
    # process has already checked the same artifacts under the mask-open guard.
    current = validate_frozen(run_map, allow_completed_verify=True)
    for key in ('selection', 'run_map_sha256', 'launch_manifest_sha256', 'receipts', 'references', 'inputs'):
        require(current[key] == proof[key], 'Verified inputs changed before scoring')
    require(guard['open_attempts'] == 0, 'Annotation opened before verified scoring')
    out.mkdir(parents=True, exist_ok=False)
    guard['active'] = False
    metadata = {}
    for seq in proof['selection']:
        meta = Path(proof['inputs'][seq]['frame_metadata']['path'])
        require(digest(meta) == proof['inputs'][seq]['frame_metadata']['sha256'], 'Archived label metadata changed')
        metadata[seq] = read(meta)
        require(digest(meta) == proof['inputs'][seq]['frame_metadata']['sha256'], 'Archived label metadata changed during read')
    mapping = read(run_map)
    all_rows, summaries, all_frame_summaries, preprompt, paired = {}, {}, {}, {}, {}
    for seq in proof['selection']:
        frames, start, _, _ = input_data(seq, EXTENDED, REMAINING, CHECKPOINT, DETECTOR)
        require(len(metadata[seq]) == len(frames), 'Label metadata coverage differs')
        references = {arm: Path(item['path']).parent for arm, item in proof['references'][seq].items()}
        folders = {arm: Path(mapping[seq][arm]) for arm in METHODS} | references
        mask_hashes = {}
        for arm, root in folders.items():
            receipt_path = root/('completion.json' if arm in ('A', 'D') else 'receipt.json')
            expected_receipt = (proof['references'][seq][arm] if arm in ('A', 'D')
                                else proof['receipts'][seq][arm])
            require(digest(receipt_path) == expected_receipt['sha256'], 'Receipt changed before scoring')
            receipt = read(receipt_path)
            require(digest(receipt_path) == expected_receipt['sha256'], 'Receipt changed during scoring')
            records = receipt['masks'] if arm in ('A', 'D') else receipt['records']
            mask_hashes[arm] = {row['frame']: row['sha256' if arm in ('A', 'D') else 'mask_sha256'] for row in records}
        rows = []
        for i, frame in enumerate(frames):
            require(digest(frame.image_path) == proof['inputs'][seq]['images'][i], 'Image changed before scoring')
            require(digest(frame.mask_path) == metadata[seq][i]['label_sha256'], 'Label changed')
            truth = load_label_mask(frame.mask_path, [1]) == 1
            require(digest(frame.mask_path) == metadata[seq][i]['label_sha256'], 'Label changed during decode')
            row = {'frame': i, 'source_frame_index': frame.frame_index,
                   'post_prompt': i > start, 'initialization_frame': i == start}
            for arm, root in folders.items():
                path = root/f'{i:06d}.png'
                require(digest(path) == mask_hashes[arm][i], 'Prediction changed before decode')
                with Image.open(path) as image:
                    prediction = np.asarray(image.convert('L')) > 0
                require(digest(path) == mask_hashes[arm][i], 'Prediction changed during decode')
                row[arm] = metric(prediction, truth)
            require(digest(frame.image_path) == proof['inputs'][seq]['images'][i], 'Image changed during scoring')
            rows.append(row)
        all_rows[seq] = rows
        summaries[seq] = {arm: summary(rows, arm) for arm in folders}
        all_frame_summaries[seq] = {arm: summary(rows, arm, post_only=False) for arm in folders}
        label_arm = 'A' if 'A' in folders else 'native'
        preprompt[seq] = {'frames': start, 'labeled_present': sum(r[label_arm]['present'] for r in rows[:start]),
                          'present_misses': {arm: sum(r[arm]['present'] and r[arm]['dice'] == 0 for r in rows[:start])
                                             for arm in folders}}
        paired[seq] = ({arm: paired_summary(rows, arm, 'A') for arm in folders if arm != 'A'}
                       if 'A' in folders else {})
    lines = ['# Additional author-system exploratory comparison', '',
             'All methods use the same archived automatic prompt and source frames. '
             'SAM2Long is offline and uses the full clip. Previously inspected patients; not a blind test.', '',
             '| Sequence | Present | Project Lean A | Refiner D | Native SAM2 | SAMURAI | SAM2Long |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for seq in proof['selection']:
        row = summaries[seq]
        values = [row[a]['mean_present_dice'] if a in row else None for a in ('A', 'D', 'native', 'samurai', 'sam2long')]
        formatted = ['N/A' if value is None else f'{value:.4f}' for value in values]
        lines.append(f"| {seq} | {row['native']['present_frames']} | " + ' | '.join(formatted) + ' |')
    lines += ['', '| Sequence | Pre-prompt present / frames | Empty alarms A / D / native / SAMURAI / SAM2Long |',
              '|---|---:|---|']
    for seq in proof['selection']:
        row, pre = summaries[seq], preprompt[seq]
        alarms = [f"{row[a]['false_alarms']}/{row[a]['empty_frames']}" if a in row else 'N/A'
                  for a in ('A', 'D', 'native', 'samurai', 'sam2long')]
        lines.append(f"| {seq} | {pre['labeled_present']}/{pre['frames']} | " + ' / '.join(alarms) + ' |')
    lines += ['', 'The project Lean baseline is a modified token-window variant, not author Lean. '
              'SAMURAI retains author thresholds. Labeled-empty is not necessarily clinical absence. '
              'The full-result rows and all-frame summaries, including the initialization frame, are in results.json. '
              'All pre-prompt outputs are empty; no new detector or redetection is used. '
              'A/D are unavailable for refiner training/validation sequences until separate full-video runs; '
              'those results will be in-sample or selection-exposed, not independent.']
    report = out/'report.md'
    report.write_text('\n'.join(lines) + '\n')
    result = {'passed': True, 'scope': 'Exploratory previously inspected patients; not independent validation.',
              'verification_sha256': digest(verification), 'run_map_sha256': digest(run_map),
              'scorer_sha256': digest(Path(__file__)), 'report_sha256': digest(report),
              'prelabel_guard': dict(guard), 'summaries': summaries, 'all_frame_summaries': all_frame_summaries,
              'preprompt': preprompt, 'paired_descriptive': paired, 'rows': all_rows}
    (out/'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'sequences': len(all_rows)}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('verify', 'score'), required=True)
    p.add_argument('--run-map', type=Path, required=True)
    p.add_argument('--verification', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.mode == 'verify':
        guard = install_mask_open_guard(DATA)
        result = validate_frozen(a.run_map)
        require(guard['open_attempts'] == 0, 'Annotation-mask open attempted during verification')
        result.update(passed=True, scope='Label-free full prediction verification.',
                      scorer_sha256=digest(Path(__file__)), label_guard=dict(guard))
        out = new_output(a.output)
        out.mkdir(parents=True, exist_ok=False)
        (out/'verification.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'passed': True, 'sequences': len(result['receipts'])}))
    else:
        if a.verification is None:
            raise ValueError('A separate label-free verification receipt is required')
        score(a.run_map, a.verification, a.output)


if __name__ == '__main__':
    main()
