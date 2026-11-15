"""Freeze label-stratified diagnostic coverage and native-cache provenance."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from evaluation.causal_memory_schedule import fixed_causal_schedule
from scripts.memory_token_io import digest, prompt_index, validate_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path,
                        default=ROOT / 'protocols/author_matching_causal_cohort_v2.json')
    args = parser.parse_args()
    destination = validate_output(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    protocol_path = args.protocol
    protocol = json.loads(protocol_path.read_text())
    archive = Path(protocol['native_cache_completion'])
    cache = json.loads(archive.read_text())
    if not cache.get('passed') or cache.get('scope') != 'cohort':
        raise RuntimeError('Archived native cohort did not complete')
    manifest = ROOT / protocol['manifest']
    if digest(manifest) != protocol['manifest_sha256']:
        raise RuntimeError('Manifest changed')
    videos = {**load_manifest(manifest, 'dev'), **load_manifest(manifest, 'test')}
    sequences = {}
    for name in protocol['sequences']:
        frames = videos[name]
        folder = Path(cache['entries'][name])
        receipt_path = folder / 'receipt.json'
        receipt = json.loads(receipt_path.read_text())
        for field in ('checkpoint_sha256', 'manifest_sha256', 'model_config'):
            if receipt[field] != protocol[field]:
                raise RuntimeError(f'Native bank provenance differs: {name}/{field}')
        prompt = prompt_index(frames, {'initialization': 'first_labeled_frame'})
        if receipt['prompt'] != prompt or receipt['num_frames'] != len(frames) or receipt['object_id'] != 1:
            raise RuntimeError(f'Native bank coverage differs: {name}')
        presence = [bool((load_label_mask(frame.mask_path, [1]) == 1).any()) for frame in frames]
        schedule = fixed_causal_schedule(presence, prompt)
        sequences[name] = {'cache_directory': str(folder), 'receipt_sha256': digest(receipt_path),
            'bank_sha256': receipt['bank_sha256'], 'schedule': schedule,
            'annotation_sha256': {str(index): digest(frame.mask_path) for index, frame in enumerate(frames)}}
    counts = {'sequences': len(sequences),
        'targets': sum(len(v['schedule']['targets']) for v in sequences.values()),
        'present_targets': sum(row['present'] for v in sequences.values() for row in v['schedule']['targets']),
        'gt_empty_targets': sum(not row['present'] for v in sequences.values() for row in v['schedule']['targets']),
        'reappearance_events': sum(len(v['schedule']['return_events']) for v in sequences.values())}
    if counts['gt_empty_targets'] == 0 or counts['reappearance_events'] == 0:
        raise RuntimeError('Requested empty/reappearance coverage is missing')
    payload = {'protocol': protocol, 'protocol_sha256': digest(protocol_path),
               'source_hashes': {p: digest(ROOT / p) for p in protocol['source_files']},
               'native_cache_completion_sha256': digest(archive), 'sequences': sequences, 'counts': counts}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps({'plan': str(destination), 'sha256': digest(destination), 'counts': counts}, indent=2))


if __name__ == '__main__':
    main()
