#!/usr/bin/env python3
"""Independently rescore seq23 viewport masks and verify saved evidence."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from modeling.seq23_viewport import VIEWPORT
from scripts.memory_token_io import digest

def read(path): return json.loads(path.read_text())

def main():
    p = argparse.ArgumentParser()
    for name in ('run', 'reference', 'gate', 'output'): p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    done = read(a.run / 'completion.json')
    old_done = read(a.reference / 'completion.json')
    if not done['passed'] or not old_done['passed'] or digest(a.reference / 'records.json') != old_done['records_sha256']:
        raise ValueError('Run or reference incomplete')
    for key, name in [('frames_sha256','frames.json'), ('setup_sha256','setup.json'), ('prompt_sha256','prompt.json'), ('positions_sha256','positions.json'), ('selections_sha256','selections.json')]:
        if digest(a.run / name) != done[key]: raise ValueError('Candidate receipt changed')
    gate = read(a.gate / 'gate.json')
    if read(a.run / 'setup.json') != gate['setup']: raise ValueError('Gate identity changed')
    for name, checksum in gate['files'].items():
        if digest(a.gate / name) != checksum: raise ValueError('Gate changed')
        if name.endswith('.png') or name == 'prompt.json':
            if digest(a.run / name) != checksum: raise ValueError('Full run differs from gate prefix')
    frames = {**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'), **load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}['seq23']
    rows, old_rows = read(a.run/'frames.json'), read(a.reference/'frames.json')
    if len(rows) != len(frames) or len(old_rows) != len(frames): raise ValueError('Frame count changed')
    result=[]
    for i, (frame, row, old) in enumerate(zip(frames, rows, old_rows)):
        if digest(frame.image_path) != row['image_sha256'] or digest(frame.mask_path) != row['label_sha256']:
            raise ValueError('Image or label changed')
        truth = load_label_mask(frame.mask_path, [1]) == 1
        metrics=[]
        for path, expected in [(a.reference/f'{i:06d}.png', old['mask_sha256']), (a.run/f'{i:06d}.png', row['mask_sha256'])]:
            if digest(path) != expected: raise ValueError('Saved mask changed')
            pred=np.asarray(Image.open(path))>0
            if pred.shape != truth.shape: raise ValueError('Geometry mismatch')
            den=int(pred.sum()+truth.sum())
            metrics.append({'dice': 2*int((pred & truth).sum())/den if den else 1., 'foreground':bool(pred.any()), 'sidebar_foreground':int(pred[:,:VIEWPORT[0]].sum())})
        if metrics[1]['sidebar_foreground'] or row['present'] != bool(truth.any()) or abs(row['dice']-metrics[1]['dice'])>1e-12 or row['truth_pixels_outside_viewport'] != int(truth[:,:VIEWPORT[0]].sum()):
            raise ValueError('Candidate scores or restoration differ')
        result.append({'frame':i,'present':bool(truth.any()),'truth_pixels_outside':int(truth[:,:VIEWPORT[0]].sum()),'reference':metrics[0],'viewport':metrics[1]})
    present=[r for r in result if r['present']];empty=[r for r in result if not r['present']]
    summary={'frames':len(frames),'present_frames':len(present),'empty_frames':len(empty),
             'present_dice_reference':float(np.mean([r['reference']['dice'] for r in present])),
             'present_dice_viewport':float(np.mean([r['viewport']['dice'] for r in present])),
             'empty_false_alarms_reference':sum(r['reference']['foreground'] for r in empty),
             'empty_false_alarms_viewport':sum(r['viewport']['foreground'] for r in empty),
             'frames_with_truth_outside_viewport':sum(r['truth_pixels_outside']>0 for r in result),
             'truth_pixels_outside_viewport':sum(r['truth_pixels_outside'] for r in result),
             'run_completion_sha256':digest(a.run/'completion.json'),'reference_completion_sha256':digest(a.reference/'completion.json'),
             'gate_sha256':digest(a.gate/'gate.json'),'audit_source_sha256':digest(Path(__file__))}
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'frames.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__': main()
