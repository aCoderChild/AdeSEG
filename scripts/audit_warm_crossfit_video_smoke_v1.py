#!/usr/bin/env python3
"""Audit fold-specific warm-routing smoke outputs without reading targets."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.crossfit_observer_capture_shared_v1 import DATA, OUT, phase_artifact
from scripts.memory_token_io import digest
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.warm_crossfit_video_contract_v1 import read_bound, require

SEQUENCES = ('seq3', 'seq23')
ARMS = ('B', 'C', 'D')


def audit(root, certificate, certificate_sha):
    root = Path(root).resolve(strict=True)
    require(root.is_relative_to(OUT.resolve(strict=True)), 'Smoke batch outside experiment root')
    cert = read_bound(certificate, certificate_sha)
    require(cert.get('passed') is True and cert.get('label_guard', {}).get('open_attempts') == 0,
            'Video certificate differs')
    receipts = {}
    for seq in SEQUENCES:
        spec = cert['references'][seq]; start = spec['prompt_frame']; end = min(spec['frames'], start + 5)
        reference = read_bound(spec['reference_path'], spec['reference_sha256'])
        receipts[seq] = {}
        for arm in ARMS:
            folder = root / 'smoke' / seq / arm; file = folder / 'receipt.json'
            sha = phase_artifact(root, f'smoke_{seq}_{arm}', file); rec = read_bound(file, sha)
            require(rec.get('passed') is True and rec.get('sequence') == seq and rec.get('arm') == arm and
                    rec.get('limit') == 4 and rec.get('certificate_sha256') == certificate_sha and
                    rec.get('reference_sha256') == spec['reference_sha256'] and rec.get('prompt_frame') == start and
                    rec.get('roi_xyxy') == spec['roi_xyxy'] and rec.get('label_guard', {}).get('open_attempts') == 0 and
                    [r['frame'] for r in rec.get('rows', [])] == list(range(start, end)) and
                    [r['frame'] for r in rec.get('masks', [])] == list(range(end)), 'Smoke receipt differs')
            runtime_path = folder / 'verification/runtime_verification.json'
            runtime = read_bound(runtime_path, rec['runtime_verification_sha256'])
            require(runtime.get('passed') is True and runtime.get('certificate_sha256') == certificate_sha and
                    runtime.get('sequence') == seq and runtime.get('device') == 'mps' and
                    runtime.get('fold_checkpoint') == cert['checkpoints'][str(spec['fold'])], 'Smoke runtime differs')
            for entry in rec['masks']:
                path = folder / f"{entry['frame']:06d}.png"
                require(digest(path) == entry['sha256'], 'Smoke output mask changed')
                with Image.open(path) as image:
                    values = np.asarray(image.convert('L'))
                    require(image.format == 'PNG' and list(image.size) == spec['image_sizes'][entry['frame']] and
                            np.isin(values, (0, 255)).all() and
                            (entry['frame'] >= start or not values.any()), 'Smoke output encoding differs')
                    if seq == 'seq23':
                        require(values.shape == (720, 1280) and not values[:, :446].any(), 'Seq23 smoke ROI differs')
                require(digest(path) == entry['sha256'], 'Smoke output changed during decoding')
            traces = {}
            for name in ('positions', 'selection', 'anchor_events', 'token_window_events', 'telemetry'):
                traces[name] = read_bound(folder / f'{name}.json', rec['trace_sha256'][name])
            require([x['frame'] for x in traces['telemetry']] == list(range(start + 1, end)),
                    'Smoke telemetry coverage differs')
            if arm == 'B':
                require([x['frame'] for x in rec['native_masks']] == list(range(start, end)),
                        'Smoke native mask coverage differs')
                for item in rec['native_masks']:
                    native = folder / 'native' / f"{item['frame']:06d}.png"
                    refrow = next(x for x in reference['rows'] if x['frame'] == item['frame'])
                    ref = Path(spec['reference_path']).parent / f"{item['frame']:06d}.png"
                    require(digest(native) == item['sha256'] and digest(ref) == refrow['mask_sha256'],
                            'Smoke native or reference mask changed')
                    with Image.open(native) as a, Image.open(ref) as b:
                        require(np.array_equal(np.asarray(a.convert('L')), np.asarray(b.convert('L'))),
                                'Smoke B native mask differs from warm A')
                    require(digest(native) == item['sha256'] and digest(ref) == refrow['mask_sha256'],
                            'Smoke native or reference mask changed during decoding')
                for name in ('positions', 'selection', 'anchor_events', 'token_window_events'):
                    require(traces[name] == reference[name][:len(traces[name])], 'Smoke B native trace differs')
            else:
                require(rec['native_masks'] == [], 'Unexpected native mask recording')
            initial = folder / f'{start:06d}.png'
            reference_initial = Path(spec['reference_path']).parent / f'{start:06d}.png'
            with Image.open(initial) as a, Image.open(reference_initial) as b:
                require(np.array_equal(np.asarray(a.convert('L')), np.asarray(b.convert('L'))),
                        'Smoke conditioning output differs from warm A')
            receipts[seq][arm] = {'path': str(file), 'sha256': sha}
    return {'passed': True, 'scope': 'Label-free fold-specific warm smoke; not efficacy or full-state parity.',
            'certificate_sha256': certificate_sha, 'receipts': receipts,
            'auditor_sha256': digest(Path(__file__))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--certificate', type=Path, required=True)
    p.add_argument('--certificate-sha256', required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); out = a.output.resolve()
    require(out.is_relative_to(OUT.resolve(strict=True)) and not out.exists(), 'Invalid audit output')
    guard = install_mask_open_guard(DATA); result = audit(a.root, a.certificate, a.certificate_sha256)
    require(guard['open_attempts'] == 0, 'Annotation opened during smoke audit')
    result['label_guard'] = dict(guard); out.mkdir(parents=True, exist_ok=False)
    (out / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'output': str(out)}))


if __name__ == '__main__': main()
