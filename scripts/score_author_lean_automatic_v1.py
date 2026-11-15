#!/usr/bin/env python3
"""Freeze-check all 21 Lean trajectories before scoring any annotation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from scripts.crossfit_observer_capture_shared_v1 import archived_frame_metadata as priority_metadata, validate_plan
from scripts.memory_token_io import digest
from scripts.run_additional_author_comparator_study_v2 import CHECKPOINT, DETECTOR, EXTENDED, OUT, REMAINING
from scripts.run_author_lean_automatic_v1 import DATA, PRIORITY, SEQUENCES, identity, inputs, new_output
from scripts.run_author_lean_automatic_study_v1 import artifact, commands, source_hashes
from scripts.score_additional_author_comparators_v2 import archived_frame_metadata, install_phase_mask_guard
from scripts.score_boundary_video_factorial_v1 import metric, summary
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.verify_author_lean_automatic_gate_v1 import verify_gate


def require(value, message):
    if not value: raise ValueError(message)


def read_bound(path, sha):
    require(digest(path) == sha, 'Artifact changed before read')
    result = json.loads(Path(path).read_text())
    require(digest(path) == sha, 'Artifact changed during read')
    return result


def overall_summary(summaries):
    eligible = [s for s in summaries.values() if s['present_frames'] and s['mean_present_dice'] is not None]
    present = sum(s['present_frames'] for s in eligible)
    empty = sum(s['empty_frames'] for s in summaries.values())
    return {'patients_with_present_frames': len(eligible), 'present_frames': present,
            'patient_macro_present_dice': float(np.mean([s['mean_present_dice'] for s in eligible])) if eligible else None,
            'frame_weighted_present_dice': float(sum(s['mean_present_dice'] * s['present_frames'] for s in eligible) / present) if present else None,
            'labeled_empty_frames': empty, 'labeled_empty_false_alarms': sum(s['false_alarms'] for s in summaries.values())}


def validate_frozen(folder, completed=()):
    """Check only image/prediction data and source identities, without labels."""
    folder = Path(folder).resolve(strict=True)
    require(folder.is_relative_to(OUT.resolve(strict=True)), 'Batch outside experiment root')
    manifest_path = folder/'run_manifest.json'; manifest_sha = digest(manifest_path)
    manifest = read_bound(manifest_path, manifest_sha); phases = commands(folder)
    require(manifest.get('sequences') == list(SEQUENCES) and manifest.get('sources') == source_hashes() and
            manifest.get('identity') == identity('seq6', CHECKPOINT, DETECTOR) and
            manifest.get('phases') == [name for name, _ in phases] and
            manifest.get('label_guard', {}).get('open_attempts') == 0, 'Frozen manifest differs')
    phase_file = folder/'phases.jsonl'; phase_sha = digest(phase_file)
    rows = [json.loads(s) for s in phase_file.read_text().splitlines()]
    require(digest(phase_file) == phase_sha, 'Phase log changed while parsing')
    expected = [name for name, _ in phases if name not in ('verify', 'score', 'audit')] + list(completed)
    require([r.get('phase') for r in rows] == expected, 'All 21 trajectories must finish before label scoring')
    for row in rows:
        name = row['phase']
        require(row.get('passed') is True and row.get('artifact_sha256') == digest(artifact(folder, name)) and
                row.get('log_sha256') == digest(folder/(name+'.log')), 'Frozen phase artifact differs')
    gate_path = folder/'gate/gate.json'
    gate = verify_gate(gate_path, manifest['identity'], CHECKPOINT, DETECTOR, EXTENDED, REMAINING)
    phase_map = {r['phase']: r['artifact_sha256'] for r in rows}
    plan = validate_plan(); records = {}
    for seq in SEQUENCES:
        frames, start, _, prompt = inputs(seq, EXTENDED, REMAINING, CHECKPOINT, DETECTOR)
        path = folder/'full'/seq/'receipt.json'; sha = phase_map[f'full_{seq}']
        rec = read_bound(path, sha)
        require(rec.get('passed') is True and rec.get('full_video') is True and rec.get('sequence') == seq and
                rec.get('identity') == identity(seq, CHECKPOINT, DETECTOR) and rec.get('prompt_frame') == start and
                rec.get('prompt') == prompt and rec.get('state_sha256') == gate['state_sha256'] and
                rec.get('gate_sha256') == digest(gate_path) and rec.get('label_guard', {}).get('open_attempts') == 0,
                'Full trajectory receipt differs')
        preds = rec.get('records', []); require(len(preds) == len(frames), 'Frame coverage differs')
        for i, (frame, row) in enumerate(zip(frames, preds, strict=True)):
            mask = path.parent/f'{i:06d}.png'; mask_sha = row['mask_sha256']
            require(row.get('frame') == i and row.get('source_frame_index') == frame.frame_index and
                    row.get('image_sha256') == digest(frame.image_path) and row.get('post_prompt') == (i >= start) and
                    digest(mask) == mask_sha, 'Full trajectory input or mask differs')
            with Image.open(frame.image_path) as src, Image.open(mask) as image:
                pixels = np.asarray(image.convert('L'))
                require(image.format == 'PNG' and image.size == src.size and np.isin(pixels, (0,255)).all() and
                        (i >= start or not pixels.any()), 'Prediction geometry or pre-prompt mask differs')
            require(digest(mask) == mask_sha, 'Prediction changed during decode')
        meta = priority_metadata(plan, seq) if seq in PRIORITY else archived_frame_metadata(seq)
        records[seq] = {'receipt_sha256': sha, 'prompt_frame': start, 'prompt': prompt,
                        'images': [row['image_sha256'] for row in preds], 'frame_metadata': meta}
    return {'manifest_sha256': manifest_sha, 'gate_sha256': digest(gate_path), 'records': records}


def score(folder):
    folder = Path(folder).resolve(strict=True); out = new_output(folder/'scores')
    guard = install_phase_mask_guard(DATA)
    proof_path = folder/'verification/verification.json'; proof_sha = digest(proof_path)
    proof = read_bound(proof_path, proof_sha)
    require(proof.get('passed') is True and proof.get('scorer_sha256') == digest(Path(__file__)) and
            proof.get('label_guard', {}).get('open_attempts') == 0, 'Verification differs')
    current = validate_frozen(folder, ('verify',))
    for key in ('manifest_sha256', 'gate_sha256', 'records'):
        require(current[key] == proof[key], 'Verified artifacts changed before scoring')
    require(guard['open_attempts'] == 0, 'Label accessed before frozen verification')
    out.mkdir(parents=True); guard['active'] = False
    all_rows, summaries, all_frames, preprompt = {}, {}, {}, {}
    for seq in SEQUENCES:
        bound = proof['records'][seq]; meta = bound['frame_metadata']
        metadata = read_bound(meta['path'], meta['sha256'])
        frames, start, _, _ = inputs(seq, EXTENDED, REMAINING, CHECKPOINT, DETECTOR)
        receipt = read_bound(folder/'full'/seq/'receipt.json', bound['receipt_sha256'])
        require(len(metadata) == len(frames) == len(receipt['records']), 'Scoring coverage differs')
        rows = []
        for i, frame in enumerate(frames):
            mask_path = folder/'full'/seq/f'{i:06d}.png'; pred_sha = receipt['records'][i]['mask_sha256']
            label_sha = metadata[i]['label_sha256']; image_sha = bound['images'][i]
            require(digest(frame.mask_path) == label_sha and digest(frame.image_path) == image_sha and
                    digest(mask_path) == pred_sha, 'Scoring inputs changed')
            truth = load_label_mask(frame.mask_path, [1]) == 1
            with Image.open(mask_path) as im: prediction = np.asarray(im.convert('L')) > 0
            require(digest(frame.mask_path) == label_sha and digest(frame.image_path) == image_sha and
                    digest(mask_path) == pred_sha, 'Scoring inputs changed during decode')
            rows.append({'frame': i, 'source_frame_index': frame.frame_index, 'post_prompt': i > start,
                         'initialization_frame': i == start, 'lean_author_mps': metric(prediction, truth)})
        all_rows[seq] = rows; summaries[seq] = summary(rows, 'lean_author_mps')
        all_frames[seq] = summary(rows, 'lean_author_mps', post_only=False)
        preprompt[seq] = {'frames': start, 'labeled_present': sum(r['lean_author_mps']['present'] for r in rows[:start]),
                          'present_misses': sum(r['lean_author_mps']['present'] and r['lean_author_mps']['dice'] == 0 for r in rows[:start])}
    overall = overall_summary(summaries)
    report = ['# Exploratory automatic-prompt author Lean comparison', '',
              'Indexed MPS RoPE compatibility correction; previously inspected patients, not independent validation.', '',
              '| Sequence | Present frames | Mean present Dice | Labeled-empty alarms |', '|---|---:|---:|---:|']
    for seq in SEQUENCES:
        s = summaries[seq]; d = s['mean_present_dice']; d = 'N/A' if d is None else f'{d:.4f}'
        report.append(f"| {seq} | {s['present_frames']} | {d} | {s['false_alarms']}/{s['empty_frames']} |")
    report += ['', f"Patient-macro present Dice: {overall['patient_macro_present_dice']} across {overall['patients_with_present_frames']} eligible patients; "
               f"frame-weighted present Dice: {overall['frame_weighted_present_dice']} across {overall['present_frames']} present frames. "
               f"Labeled-empty false alarms: {overall['labeled_empty_false_alarms']}/{overall['labeled_empty_frames']}.", '',
               'Post-prompt summaries exclude initialization. Seq23 uses its retrospective crop. '
               'Seq1/7 had no automatic initialization and are not included in segmentation summaries.']
    report_path = out/'report.md'; report_path.write_text('\n'.join(report)+'\n')
    result = {'passed': True, 'verification_sha256': proof_sha, 'scorer_sha256': digest(Path(__file__)),
              'report_sha256': digest(report_path), 'prelabel_guard': dict(guard), 'rows': all_rows,
              'summaries': summaries, 'all_frame_summaries': all_frames, 'preprompt': preprompt, 'overall': overall}
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'passed': True, 'sequences': len(all_rows)}))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--mode', choices=('verify', 'score'), required=True); a = p.parse_args()
    if a.mode == 'score': score(a.run); return
    guard = install_mask_open_guard(DATA); result = validate_frozen(a.run)
    require(guard['open_attempts'] == 0, 'Annotation open during verification')
    result.update(passed=True, scorer_sha256=digest(Path(__file__)), label_guard=dict(guard))
    out = new_output(a.run/'verification'); out.mkdir(parents=True)
    (out/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'passed': True, 'sequences': len(result['records'])}))


if __name__ == '__main__': main()
