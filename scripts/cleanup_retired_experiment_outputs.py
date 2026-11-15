#!/usr/bin/env python3
"""Remove explicitly invalid attempts and retired failed-candidate tensors.

Writes a durable itemized receipt before unlinking. It does not follow symlinks
or remove reports, summaries, trained model checkpoints, or current Lean runs.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

ROOT = Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment').resolve()


def main():
    report_dir = ROOT / 'cleanup_reports'
    report_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    receipt = report_dir / f'retired_outputs_{stamp}.json'
    if receipt.exists():
        raise FileExistsError(receipt)
    # Only the two explicitly stopped ATC/PiToMe candidates; their ToMe
    # comparator tensors and all reports/results are deliberately retained.
    patterns = {
        'causal_series02': ('*_atc_global.pt', '*_atc_regional.pt'),
        'pitome_causal_series01': ('*_pitome_global.pt', '*_pitome_regional.pt'),
    }
    items = []
    for study, suffixes in patterns.items():
        parent = ROOT / 'tome_region_identification_v1' / study
        completed = json.loads((parent / 'completion.json').read_text())
        if not completed.get('complete'):
            raise ValueError(f'Study has no completed receipt: {study}')
        marker = (parent / ('status.md' if study == 'causal_series02' else 'status.json')).read_text()
        if 'fail' not in marker.lower() and 'stop' not in marker.lower():
            raise ValueError(f'Study does not document stopped candidates: {study}')
        for pattern in suffixes:
            files = sorted((parent / 'broader').glob(f'seq*/{pattern}'))
            if len(files) != 318:
                raise ValueError(f'Unexpected file count in {study}/{pattern}: {len(files)}')
            for path in files:
                if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(parent.resolve()):
                    raise ValueError(f'Unsafe file: {path}')
                stat = path.stat()
                items.append({'path': str(path.relative_to(ROOT)), 'kind': 'retired_failed_candidate_tensor',
                              'logical_bytes': stat.st_size, 'allocated_bytes': stat.st_blocks * 512})
    marker_globs = [
        'lean_auto_multiscale_v1/gate_attempt*/aborted.json',
        'prompt_feature_refinement_v1/*/aborted.json',
        'prompt_feature_refinement_v1/*/superseded_invalid.json',
        'sasvi_samrefiner_20261114/*/superseded_invalid.json',
    ]
    invalid_dirs = []
    for pattern in marker_globs:
        for marker in sorted(ROOT.glob(pattern)):
            parent = marker.parent
            if parent.is_symlink() or not parent.resolve().is_relative_to(ROOT):
                raise ValueError(f'Unsafe invalid directory: {parent}')
            invalid = json.loads(marker.read_text())
            if not (invalid.get('invalid') is True or invalid.get('valid_for_requested_study') is False or invalid.get('completed') is False):
                raise ValueError(f'Unrecognized invalid marker: {marker}')
            logical = allocated = files = 0
            for base, dirs, filenames in os.walk(parent, followlinks=False):
                for name in filenames:
                    path = Path(base) / name
                    if path.is_symlink(): continue
                    stat = path.stat(); logical += stat.st_size; allocated += stat.st_blocks * 512; files += 1
            items.append({'path': str(parent.relative_to(ROOT)), 'kind': 'invalid_or_aborted_attempt',
                          'logical_bytes': logical, 'allocated_bytes': allocated, 'files': files, 'marker': invalid})
            invalid_dirs.append(parent)
    payload = {'created_utc': stamp, 'state': 'deletion_in_progress',
               'scope': 'Explicitly stopped ATC/PiToMe tensor files and explicitly invalid/aborted attempts; current Lean results and comparison artifacts retained.',
               'warning': 'Deleted items are removed from Google Drive as well as the local cache. Raw tensor-level re-audit of the retired candidates is no longer possible without regeneration.',
               'items': items, 'logical_bytes_planned': sum(x['logical_bytes'] for x in items),
               'allocated_bytes_planned': sum(x['allocated_bytes'] for x in items)}
    receipt.write_text(json.dumps(payload, indent=2) + '\n')
    deleted = []
    try:
        for item in items:
            path = ROOT / item['path']
            if item['kind'] == 'retired_failed_candidate_tensor':
                path.unlink()
            else:
                shutil.rmtree(path)
            deleted.append(item['path'])
        payload['state'] = 'complete'
    except Exception as exc:
        payload['state'] = 'failed'
        payload['error'] = repr(exc)
        raise
    finally:
        payload['deleted_paths'] = deleted
        receipt.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'receipt': str(receipt), 'state': payload['state'], 'deleted_items': len(deleted),
                      'logical_gib': payload['logical_bytes_planned'] / 2**30,
                      'allocated_gib': payload['allocated_bytes_planned'] / 2**30}, indent=2))


if __name__ == '__main__': main()
