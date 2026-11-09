"""Freeze outcome-independent TRAIN cases before memory-value oracle inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from scripts.mavos_value_utils import atomic_json
from scripts.memory_token_io import digest, prompt_index, validate_output
from scripts.run_token_conditioned_key_adapter import CHECKPOINT_DEFAULT, MANIFEST, PROTOCOL

DESIGN = ROOT / 'protocols/mask_memory_value_headroom_v1.json'


def target_indices(length):
    if length < 8:
        return []
    return sorted({7 + (length - 8) * i // 2 for i in range(3)})


def run(out):
    out = validate_output(out)
    out.mkdir(parents=True, exist_ok=False)
    design, parent = json.loads(DESIGN.read_text()), json.loads(PROTOCOL.read_text())
    names = design['data']['sequences']
    if names != parent['data']['train_sequences'] or design['data']['validation_access'] or design['data']['test_access']:
        raise RuntimeError('Headroom case plan must use exactly the existing TRAIN sequences')
    if digest(MANIFEST) != parent['data']['manifest_sha256']:
        raise RuntimeError('Frozen manifest changed')
    if digest(CHECKPOINT_DEFAULT) != parent['model']['checkpoint_sha256']:
        raise RuntimeError('Pretrained checkpoint changed')
    videos = load_manifest(MANIFEST, 'dev')
    cases, sequences = [], {}
    for name in names:
        original = videos[name]
        start = prompt_index(original, {'initialization': 'first_labeled_frame'})
        frames = original[start:]
        targets = target_indices(len(frames))
        sequences[name] = {'prompt_manifest_offset': start, 'prompt_source_frame': original[start].frame_index,
            'post_prompt_reindexed_length': len(frames), 'target_indices': targets,
            'skip_reason': None if targets else 'too_short_for_prompt_plus_six_recent_frames'}
        for target in targets:
            cases.append({'id': f'{name}_t{target}', 'sequence': name, 'target_local_frame': target,
                'target_source_frame': frames[target].frame_index,
                'target_image': str(frames[target].image_path), 'target_mask': str(frames[target].mask_path),
                'prefix_frames': target + 1})
    identity = {'protocol_sha256': digest(DESIGN), 'parent_protocol_sha256': digest(PROTOCOL),
        'manifest_sha256': digest(MANIFEST), 'checkpoint_sha256': digest(CHECKPOINT_DEFAULT),
        'source_sha256': {p: digest(ROOT / p) for p in (
            'scripts/prepare_mask_memory_headroom.py', 'tests/test_mask_memory_headroom_plan.py',
            'scripts/mask_memory_teacher.py', 'modeling/memory_value_oracle.py',
            'modeling/mavos_focal_value_modulation.py', 'modeling/native_memory_layout.py')}}
    result = {'identity': identity, 'sequences': sequences, 'cases': cases,
        'selection_uses_outcomes': False, 'inference_started': False,
        'validation_accessed': False, 'test_accessed': False, 'efficacy_authorized': False}
    atomic_json(out / 'protocol_frozen.json', design)
    atomic_json(out / 'cases.json', result)
    atomic_json(out / 'completion.json', {'passed': True, 'cases_sha256': digest(out / 'cases.json'),
        'scope': 'Case planning only; no native or oracle inference', 'efficacy_authorized': False})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps({'planned_cases': len(result['cases']), 'sequences': result['sequences']}, indent=2))
